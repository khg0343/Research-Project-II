from .aihub_eval import do_aihub_evaluation
import logging

def aihub_evaluation(
    dataset,
    predictions,
    output_folder, **_
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("performing aihub evaluation")
    return do_aihub_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger
    )
