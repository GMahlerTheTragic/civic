import math
import os.path
from civic.config import MODEL_CHECKPOINT_DIR
from civic.training.ModelTrainer import ModelTrainer


class MlmRobertaLongPreTrainingModelTrainer(ModelTrainer):
    def __init__(self, trainer):
        self.trainer = trainer

    def do_model_training(self, n_epochs=None):
        eval_loss = self.trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(f"Initial eval bpc: {eval_loss/math.log(2)}")
        print("Pretraining... ")
        self.trainer.train()
        eval_loss = self.trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")
        self.trainer.save_model(os.path.join(MODEL_CHECKPOINT_DIR, "pretrained"))
