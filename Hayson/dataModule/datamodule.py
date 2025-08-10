        self.tr        self.train_dataset = None
        self.val_dataset = None
        self.val_df = None

    def setup(self, stage: str = None):
        # splits the data into training and validation
        train_df = self.feature_df[self.feature_df[self.date_col] < self.split_date]
        self.val_df = self.feature_df[self.feature_df[self.date_col] >= self.split_date]aset = None
        self.val_dataset = None
        self.train_df = None
        self.val_df = None

    def setup(self, stage: str = None):
        # splits the data into training and validation
        def __init__(self, feature_df: pd.DataFrame, split_date: str, batch_size: int, date_col: str, target_col: str):
        super().__init__()
        self.feature_df = feature_df
        self.split_date = split_date
        self.batch_size = batch_size
        self.date_col = date_col
        self.target_col = target_col
        self.train_dataset = None
        self.val_dataset = None
        self.train_df = None
        self.val_df = None

    def setup(self, stage: str = None):
        # splits the data into training and validation
        self.train_df = self.feature_df[self.feature_df[self.date_col] < self.split_date]
        self.val_df = self.feature_df[self.feature_df[self.date_col] >= self.split_date]
        self.val_df = None

    def setup(self, stage: str = None):
        # splits the data into training and validation
        train_df = self.feature_df[self.feature_df[self.date_col] < self.split_date]
        val_df = self.feature_df[self.feature_df[self.date_col] >= self.split_date]
        self.val_df = val_df
        self.val_df = val_df
        self.val_df = val_df
        self.val_df = val_df
        self.val_df = val_df
