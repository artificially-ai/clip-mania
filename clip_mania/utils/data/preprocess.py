import os


class DatasetProcessor:

    @staticmethod
    def create_dataset(dataset_path: str) -> dict:
        dataset = {"items": []}
        for root, dirs, _ in os.walk(dataset_path):
            dirs.sort()
            for label in dirs:
                for _, _, sub_files in os.walk(os.path.join(root, label)):
                    for file in sub_files:
                        dataset["items"].append((os.path.join(root, label, file),
                                                f"This is a picture of a(n) {label}."))

        return dataset

    @staticmethod
    def create_indexed_prompts(dataset_path: str) -> dict:
        prompts = {}
        label_value = 0
        for root, dirs, _ in os.walk(dataset_path):
            dirs.sort()
            for label in dirs:
                prompts[f"This is a picture of a(n) {label}."] = label_value
                label_value += 1

        return prompts
