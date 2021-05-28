import os


class DatasetProcessor:

    @staticmethod
    def create_dataset(dataset_path: str):
        dataset = {'item': []}
        for root, dirs, _ in os.walk(dataset_path):
            for label in dirs:
                for _, _, sub_files in os.walk(os.path.join(root, label)):
                    for file in sub_files:
                        dataset['item'].append((os.path.join(root, label, file),
                                                f'This is a picture of a {label}.'))

        return dataset
