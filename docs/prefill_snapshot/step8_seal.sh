#!/bin/bash
# step8_seal.sh -- 验证填充范围、计算三版 SHA、回填自指 SHA、生成 diff、追加 ledger
# 不执行任何 git 写操作。
set -e
cd /root/mamba-scan-study

ADD=MAIN_PREREG_ADDENDUM_03_CONTINGENCY.md
CD=CODE_DELTA_68dff0b_32edce6.md
SNAP=docs/prefill_snapshot
V0_ADD=7f02f9ba8c0a8f02708cabb048dce12c6f9b001a6fe3758a6b3c7a02e64c2beb
V0_CD=9918623eaed4a1dcec7efe0acd03d78a67302681e5bf331c8f2589e9d79400d4

fail() { echo; echo "!!! $1"; echo "!!! 已停止，未做任何修改。"; exit 1; }

echo "===== 1. V0 快照完整性 ====="
a=$(sha256sum $SNAP/ADDENDUM_03.prefill.md | cut -d' ' -f1)
b=$(sha256sum $SNAP/CODE_DELTA.prefill.md  | cut -d' ' -f1)
[ "$a" = "$V0_ADD" ] || fail "ADDENDUM_03 快照已被改动"
[ "$b" = "$V0_CD"  ] || fail "CODE_DELTA 快照已被改动"
echo "  两份 V0 快照与时间戳记录一致"

echo
echo "===== 2. 填充范围检查 ====="
grep -q "d_model = 256" $ADD || fail "ADDENDUM_03 §8 似乎未填入 d_model"
grep -q "282,122" $ADD       || fail "ADDENDUM_03 §8 似乎未填入参数量"
if grep -q "落盘前须核实" $ADD; then fail "ADDENDUM_03 §8 的回填指令引用块仍在，应删除"; fi
grep -q "未查看结果的声明" $ADD || fail "ADDENDUM_03 开头的未查看结果声明缺失"
grep -q "^## 5.2 " $CD       || fail "CODE_DELTA §5.2 缺失"
n=$(sed -n '/^## 5.2 /,$p' $CD | grep -c '^| ')
[ "$n" -ge 16 ] || fail "CODE_DELTA §5.2 的表格行数异常（$n），疑似截断"
echo "  ADDENDUM_03 §8 已填、指令块已删、声明段保留"
echo "  CODE_DELTA §5.2 存在，表格行数 $n"

echo
echo "===== 3. 生成 diff 并显示改动区间 ====="
diff -u $SNAP/ADDENDUM_03.prefill.md $ADD > $SNAP/ADDENDUM_03.fill.diff || true
diff -u $SNAP/CODE_DELTA.prefill.md  $CD  > $SNAP/CODE_DELTA.fill.diff  || true
echo "--- ADDENDUM_03 hunk ---"; grep '^@@' $SNAP/ADDENDUM_03.fill.diff || echo "  (无差异)"
echo "--- CODE_DELTA hunk ---";  grep '^@@' $SNAP/CODE_DELTA.fill.diff  || echo "  (无差异)"
echo "--- 改动行数（+/-）---"
printf "  ADDENDUM_03: %s\n" "$(grep -c '^[+-][^+-]' $SNAP/ADDENDUM_03.fill.diff || true)"
printf "  CODE_DELTA : %s\n" "$(grep -c '^[+-][^+-]' $SNAP/CODE_DELTA.fill.diff || true)"

echo
echo "===== 4. V1 SHA（填充后、自指回填前）====="
V1_ADD=$(sha256sum $ADD | cut -d' ' -f1)
V1_CD=$(sha256sum $CD  | cut -d' ' -f1)
echo "  ADDENDUM_03 V1: $V1_ADD"
echo "  CODE_DELTA  V1: $V1_CD"

echo
echo "===== 5. 回填 ADDENDUM_03 自指 SHA ====="
if grep -q '本件 SHA：\* `<落盘后回填>`' $ADD || grep -q '<落盘后回填>' $ADD; then
  sed -i "s|<落盘后回填>|$V1_ADD|" $ADD
  echo "  已回填 V1 SHA"
else
  echo "  占位符不存在（可能已回填），跳过"
fi
V2_ADD=$(sha256sum $ADD | cut -d' ' -f1)
echo "  ADDENDUM_03 V2: $V2_ADD"

echo
echo "===== 6. 冻结 split 文件实测 SHA（erratum 8g.6 用）====="
SPLIT_SHA=$(sha256sum P0B_CIFAR10_VAL_SPLIT_FROZEN.json | cut -d' ' -f1)
echo "  $SPLIT_SHA"

echo
echo "===== 7. 追加 ledger ====="
[ -f /root/mamba-scan-study/ledger_append.md ] || fail "ledger_append.md 不在仓库根目录，请先上传"
sed -e "s|{{V0_ADD}}|$V0_ADD|"   -e "s|{{V0_CD}}|$V0_CD|" \
    -e "s|{{V1_ADD}}|$V1_ADD|"   -e "s|{{V1_CD}}|$V1_CD|" \
    -e "s|{{V2_ADD}}|$V2_ADD|"   -e "s|{{SPLIT_SHA}}|$SPLIT_SHA|" \
    ledger_append.md >> docs/03_EVIDENCE_LEDGER.md
rm ledger_append.md
if grep -q '{{' docs/03_EVIDENCE_LEDGER.md; then fail "ledger 中仍有未替换的占位符"; fi
echo "  已追加，占位符全部替换"

echo
echo "===== 8. 待 commit 清单 ====="
git status --short --untracked-files=all

echo
echo "===== 完成。未执行任何 git 写操作。 ====="
echo "请核对上面第 3 节的 hunk 区间：ADDENDUM_03 应只有一处（§8 附近），"
echo "CODE_DELTA 应只有一处（文件末尾追加）。确认后再跑 commit。"
