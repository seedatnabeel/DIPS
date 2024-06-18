class DataHandler:
    def __init__(self, dataloader, unshuffled_dataloader) -> None:
        self.dataloader = dataloader
        self.dataloader_unshuffled = unshuffled_dataloader

    def toJson(self) -> str:
        import json

        """
        Returns a JSON representation of the DataHandler object.

        Returns:
            str: A JSON representation of the DataHandler object in the form of a string.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
