import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from dataModule import get_data_loader_with_module

# 1. Data Loading
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
start_date = '2020-01-01'
end_date = '2023-01-01'
encoder_len = 90
predict_len = 21
batch_size = 128

import dotenv
dotenv.load_dotenv()
news_api_key=os.getenv('NEWS_API_KEY')
fred_api_key=os.getenv('FRED_API_KEY')
api_ninjas_key=os.getenv('API_NINJAS_KEY')

print(f"NEWS_API_KEY: {news_api_key}"
      f"\nFRED_API_KEY: {fred_api_key}"
        f"\nAPI_NINJAS_KEY: {api_ninjas_key}")
print(f"Symbols: {symbols}")

dataloader, datamodule = get_data_loader_with_module(
    symbols=symbols,
    start=start_date,
    end=end_date,
    encoder_len=encoder_len,
    predict_len=predict_len,
    batch_size=batch_size,
    news_api_key=os.getenv('NEWS_API_KEY'),
    fred_api_key=os.getenv('FRED_API_KEY'),
    api_ninjas_key=os.getenv('API_NINJAS_KEY')
)

# 2. Model Definition
# Using the TemporalFusionTransformer from pytorch_forecasting
model = TemporalFusionTransformer.from_dataset(
    datamodule.train_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=torch.nn.MSELoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# 3. Training
# Callbacks
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(
    dirpath="tft_checkpoints",
    filename="best-model",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

# Trainer
trainer = Trainer(
    max_epochs=100,
    accelerator="auto",
    callbacks=[early_stop_callback, checkpoint_callback],
)

# Fit the model
trainer.fit(
    model,
    train_dataloaders=datamodule.train_dataloader(),
    val_dataloaders=datamodule.val_dataloader(),
)

# 4. Evaluation
# Load the best model
best_model_path = checkpoint_callback.best_model_path
if best_model_path:
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calculate mean absolute error on validation set
    actuals = torch.cat([y[0] for x, y in iter(datamodule.val_dataloader())])
    predictions = best_tft.predict(datamodule.val_dataloader())
    print(f"MAE: {(actuals - predictions).abs().mean()}")

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(datamodule.val_dataloader(), mode="raw", return_x=True)

    # Plotting example
    # best_tft.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True);
else:
    print("No best model found. Training might have failed or not run.")
