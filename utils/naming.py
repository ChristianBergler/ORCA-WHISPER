import random


class OrcaSpotNameGenerator:
    def __init__(self,
                 start_idx=1,
                 start_year=1960,
                 tape_start=1,
                 sample_size=1280,
                 sample_min=1e8,
                 sample_max=9e8,
                 tape_min=1,
                 tape_max=999,
                 max_year=2015,
                 inc_year=True,
                 ext=".png",
                 ):
        self.start_idx = start_idx
        self.start_year = start_year
        self.tape_start = tape_start
        self.sample_size = sample_size
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.tape_min = tape_min
        self.tape_max = tape_max
        self.max_year = max_year
        self.inc_year = inc_year
        self.ext = ext

        self.current_idx = start_idx
        self.current_year = start_year
        self.current_tape = tape_start
        self.current_tape_side = "A"

    def _update_state(self):
        self.current_idx += 1
        self.current_tape = random.randint(self.tape_min, self.tape_max)
        if self.inc_year:
            self.current_year += 1
            if self.current_year >= self.max_year:
                self.current_year = self.start_year

        self.current_tape_side = "B" if self.current_tape_side == "A" else "A"

    def generate(self, class_name):
        tape_str = str(self.current_tape).zfill(3) + self.current_tape_side
        pos_start = random.randint(int(self.sample_min), int(self.sample_max - self.sample_size))
        pos_end = int(pos_start + self.sample_size)
        name = f"{class_name}_{self.current_idx}_{self.current_year}_{tape_str}_{pos_start}_{pos_end}{self.ext}"
        self._update_state()
        return name
