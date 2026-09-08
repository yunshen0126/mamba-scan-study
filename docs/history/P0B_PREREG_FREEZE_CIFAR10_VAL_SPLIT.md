# P0-B CIFAR-10 Validation Split Freeze

Before P0-B performance runs, the official CIFAR-10 `train=True` split is frozen into 45,000 train and 5,000 validation indices. A single `np.random.Generator(np.random.PCG64(20260720))` is used continuously over classes 0 through 9; each class contributes 4,500 train and 500 validation members, after which both arrays are sorted.

The frozen arrays in `P0B_CIFAR10_VAL_SPLIT_FROZEN.json` are the only runtime source of truth. Train uses the existing RandomCrop/RandomHorizontalFlip transform; validation uses the existing deterministic eval transform through two `train=True` dataset instances. Official `train=False` is not instantiated or evaluated in P0-B. A download archive may physically contain test files; code does not construct or read a test dataset.

Frozen JSON SHA-256: `e28719c9154bfcdce9c89ab5c91529eb27403ce54483eac494708c0f072b1f09`.

Split generation RNG and training RNG are separate. Membership is never adjusted from model performance. The split SHA is required in every future P0-B metadata/checkpoint.
