import logging
from omegaconf import DictConfig
from .analysis.analysis_main import Analyzer
from .utils import print_config
import hydra

__all__ = ['log', 'analysis', ]


log = logging.getLogger(__name__)


def analysis(config: DictConfig):
    print_config(
        config,
        fields=(
            "globals",
            "classify",
            "extract",
            "plotter",
        ),
        resolve=False
    )
    log.info("Start analysis.")
    output_file = None
    if "output_path" in config.globals:
        output_file = config.globals.output_path.replace(config.globals.output_path.split(".")[-1], "")[:-1]
    if "extract" in config:
        if config.extract:
            log.info("Get extract.")
            extracter = hydra.utils.instantiate(config.extract)
    else:
        extracter = None
    if "classify" in config:
        if config.classify:
            log.info("Get classify.")
            classifier = hydra.utils.instantiate(config.classify)
    else:
        classifier = None
    if "plotter" in config:
        if config.plotter:
            log.info("Get plotter.")
            plotter = hydra.utils.instantiate(config.plotter)
    else:
        plotter = None

    log.info("Initalize analysis.")
    analyzer = Analyzer(extracter=extracter, classifier=classifier, plotter=plotter)
    log.info("Analysing....")
    if "input_path" in config.globals:
        analyzer.analysis(output_file=output_file, input_path=config.globals.input_path)
    else:
        analyzer.analysis(output_file=output_file)
    log.info("Done.")

