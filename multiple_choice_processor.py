from torchblocks.processor.base import DataProcessor


class MultipleChoiceProcessor(DataProcessor):
    """
        多选题
    """
    def convert_to_features(self, examples, label_list, max_seq_length):
        pass