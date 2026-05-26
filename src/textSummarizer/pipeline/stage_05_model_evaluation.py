from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>> Model Evaluation Stage Started <<<<")

            config = ConfigurationManager()

            model_eval_config = config.get_model_evaluation_config()

            model_evaluator = ModelEvaluation(config=model_eval_config)

            model_evaluator.evaluate()

            logger.info(">>>> Model Evaluation Stage Completed <<<<")

        except Exception as e:
            logger.exception(e)
            raise e