from helper_functions import create_dataset

if __name__ == '__main__':
    create_dataset("data/snirf/bci_task_1_arms_talk.snirf",
                   0, "data/datasets/arms_talk_")
    create_dataset("data/snirf/bci_task_1_arms_talk.snirf",
                   10, "data/datasets/arms_talk_")
    create_dataset("data/snirf/bci_task_1_arms_talk.snirf",
                   50, "data/datasets/arms_talk_")
    create_dataset("data/snirf/bci_task_2_arithmetic_audiobook.snirf",
                   0, "data/datasets/arithmetic_audiobook_")
    create_dataset("data/snirf/bci_task_2_arithmetic_audiobook.snirf",
                   10, "data/datasets/arithmetic_audiobook_")
    create_dataset("data/snirf/bci_task_2_arithmetic_audiobook.snirf",
                   50, "data/datasets/arithmetic_audiobook_")
    create_dataset("data/snirf/bci_task_3_arithmetic_rotation.snirf",
                   0, "data/datasets/arithmetic_rotation_")
    create_dataset("data/snirf/bci_task_3_arithmetic_rotation.snirf",
                   10, "data/datasets/arithmetic_rotation_")
    create_dataset("data/snirf/bci_task_3_arithmetic_rotation.snirf",
                   50, "data/datasets/arithmetic_rotation_")
