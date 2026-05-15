import random
class PreferenceBuffer:
    def __init__(self,
                 capacity=100000):
        self.capacity = capacity
        self.data = []

    def add(self,
            seg1,
            seg2,
            label):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append((seg1, seg2, label))

    def sample(self,
               batch_size):
        batch = random.sample(self.data, batch_size)
        return batch

    def __len__(self):
        return len(self.data)