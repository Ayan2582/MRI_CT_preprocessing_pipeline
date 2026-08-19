"""
image_pool.py
─────────────
A buffer of previously generated images, shown to the discriminator instead of
only the freshest ones.

WHAT IT FIXES. Without it, D at step t only ever sees G's output at step t. The
two then chase each other: G finds a texture that fools the current D, D learns
to reject exactly that texture, G moves on, and D immediately forgets the texture
it just learned to reject because no example of it will ever appear again. The
result is an oscillation neither player escapes, and it is the failure the
CycleGAN paper attributes this buffer to fixing.

HOW IT WORKS. Keep the last `pool_size` fakes. For each image in a batch, with
probability 1/2 hand D the new one and store it; otherwise hand D a random stored
one and put the new one in its place. So D is trained against a moving window of
G's recent history rather than a single instant of it.

WHY pix2pix DOES NOT NEED THIS. There, L1 pins the output to a specific target,
so G cannot wander far between steps and there is little history to forget.
CycleGAN has no such anchor — nothing says what the output should look like pixel
by pixel — which is exactly why the oscillation has room to develop.

NOT CHECKPOINTED, DELIBERATELY. The pool is a few dozen images of transient
training state, and restoring it would mean carrying ~50 tensors in every
checkpoint for a buffer that refills in well under an epoch. The bit-exact resume
check in scripts/smoke_test.py therefore only holds for models that have no pool;
CHECK 6 asserts CycleGAN's structure and step instead.
"""

import random

import torch


class ImagePool:
    """
    Parameters
    ----------
    pool_size : how many past images to retain. 50 is the CycleGAN default and
                is roughly "the last few batches" at the batch sizes used here.
                0 disables the buffer entirely, which makes query() the identity
                and is the right setting for an ablation.
    """

    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.images = []

    def __len__(self):
        return len(self.images)

    def query(self, images):
        """
        Swap some of `images` for stored ones, and return the batch to show D.

        Always detached. A stored tensor that still carried its graph would keep
        the generator's activations from several steps ago alive — a slow memory
        leak that ends in an OOM tens of epochs in, long after the change that
        caused it.
        """
        if self.pool_size == 0:
            return images

        out = []
        for image in images:
            image = image.detach().unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image)
                out.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.pool_size)
                out.append(self.images[index])
                self.images[index] = image
            else:
                out.append(image)

        return torch.cat(out, dim=0)
