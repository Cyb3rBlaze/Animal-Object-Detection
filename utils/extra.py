# SUPER IMPORTANT FOR RIGHT FORMAT
def collate_fn(batch):
    return tuple(zip(*batch))