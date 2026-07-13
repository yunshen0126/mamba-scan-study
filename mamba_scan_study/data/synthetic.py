import torch
from torch.utils.data import Dataset


TASK_DIRS = ("vertical", "horizontal", "diagonal")
SIGNAL_STRENGTHS = ("line", "single_patch", "single_pixel")


class CarryDataset(Dataset):
    """
    Binary carry task with random background.

    A bit is encoded at a source location and a query marker is placed at the
    corresponding target location. Direction controls the source/target geometry.
    """

    def __init__(
        self,
        num_samples=10000,
        grid_size=8,
        patch_size=4,
        in_chans=3,
        task_dir="vertical",
        signal_strength="single_patch",
        amplitude=2.5,
        noise_std=1.0,
        shuffle_source=False,
        seed=0,
    ):
        if task_dir not in TASK_DIRS:
            raise ValueError(f"task_dir must be one of {TASK_DIRS}, got {task_dir!r}")
        if signal_strength not in SIGNAL_STRENGTHS:
            raise ValueError(
                f"signal_strength must be one of {SIGNAL_STRENGTHS}, got {signal_strength!r}"
            )
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.img_size = grid_size * patch_size
        self.in_chans = in_chans
        self.task_dir = task_dir
        self.signal_strength = signal_strength
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.shuffle_source = shuffle_source
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def _rng(self, idx):
        g = torch.Generator()
        g.manual_seed(self.seed + idx * 1009)
        return g

    def _patch_slice(self, r, c):
        p = self.patch_size
        return slice(r * p, (r + 1) * p), slice(c * p, (c + 1) * p)

    def _draw_patch_signal(self, image, r, c, value, channel, generator):
        rr, cc = self._patch_slice(r, c)
        if self.signal_strength == "single_pixel":
            pr = int(torch.randint(0, self.patch_size, (1,), generator=generator))
            pc = int(torch.randint(0, self.patch_size, (1,), generator=generator))
            image[channel, rr.start + pr, cc.start + pc] += value
        else:
            image[channel, rr, cc] += value

    def _line_source_and_target(self, generator):
        n = self.grid_size
        if self.task_dir == "vertical":
            c = int(torch.randint(0, n, (1,), generator=generator))
            return [(0, c)], (n - 1, c)
        if self.task_dir == "horizontal":
            r = int(torch.randint(0, n, (1,), generator=generator))
            return [(r, 0)], (r, n - 1)

        offset = int(torch.randint(-(n - 2), n - 1, (1,), generator=generator))
        coords = [(r, r + offset) for r in range(n) if 0 <= r + offset < n]
        if len(coords) < 2:
            coords = [(i, i) for i in range(n)]
        return [coords[0]], coords[-1]

    def __getitem__(self, idx):
        g = self._rng(idx)
        image = torch.randn(
            self.in_chans, self.img_size, self.img_size, generator=g
        ) * self.noise_std
        label = int(torch.randint(0, 2, (1,), generator=g))
        sources, target = self._line_source_and_target(g)
        primary_source = sources[0]
        source_label = int(torch.randint(0, 2, (1,), generator=g)) if self.shuffle_source else label
        bit_value = self.amplitude if source_label == 1 else -self.amplitude

        if self.signal_strength == "line":
            n = self.grid_size
            if self.task_dir == "vertical":
                sources = [(0, c) for c in range(n)]
            elif self.task_dir == "horizontal":
                sources = [(r, 0) for r in range(n)]
            else:
                sources = [(i, i) for i in range(n)]

        for r, c in sources:
            self._draw_patch_signal(image, r, c, bit_value, channel=0, generator=g)
        self._draw_patch_signal(image, target[0], target[1], self.amplitude, channel=1, generator=g)
        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(target[0], dtype=torch.long),
            torch.tensor(target[1], dtype=torch.long),
            torch.tensor(primary_source[0], dtype=torch.long),
            torch.tensor(primary_source[1], dtype=torch.long),
        )


class VerticalCarry(CarryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_dir="vertical", **kwargs)


class HorizontalCarry(CarryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_dir="horizontal", **kwargs)


class DiagonalCarry(CarryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task_dir="diagonal", **kwargs)


def build_synthetic_dataset(split, **kwargs):
    seed = kwargs.pop("seed", 0)
    if split == "train":
        return CarryDataset(seed=seed, **kwargs)
    if split in ("val", "test"):
        return CarryDataset(seed=seed + 1_000_000, **kwargs)
    raise ValueError("split must be 'train', 'val', or 'test'")
