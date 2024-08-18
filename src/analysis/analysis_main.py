import logging
from abc import ABC
from typing import Optional
from typing import Callable
import pandas as pd

__all__ = ['log', 'Analyzer', ]


log = logging.getLogger(__name__)



class Analyzer(ABC):

    def __init__(
        self,
        extracter: Optional[Callable] = None,
        classifier: Optional[Callable] = None,
        plotter: Optional[Callable] = None,
    ):
        log.info("Get all tools...")
        self.extracter = extracter
        self.classifier = classifier
        self.plotter = plotter
        log.info("Done.")

    def analysis(self, output_file: str, input_path: Optional[str] = None):
        tails = None
        if self.extracter is not None:
            log.info("extracting...")
            if hasattr(self.extracter, "tails"):
                output_files = []
                tails = self.extracter.tails
                for tail in tails:
                    output_files.append(output_file + tail)
            results = self.extracter()
            log.info("Done.")
        if self.classifier is not None:
            log.info("classifing...")
            if results is not None:
                for i, result in enumerate(results):
                    result = self.classifier(result)
                    try:
                        dataframe = pd.DataFrame.from_dict(result, orient='index').T
                        dataframe.to_csv(f'{output_files[i]}.csv')
                    except:
                        try:
                            dataframe = pd.DataFrame.from_dict({
                                (i, j):
                                result[i][j]
                                for i in result.keys()
                                for j in result[i].keys()
                            },
                                orient='index',
                            ).T
                            dataframe.to_csv(f'{output_files[i]}.csv')
                        except Exception:
                            raise
            results = None
            log.info("Done.")
        if self.plotter is not None:
            log.info("plotting...")
            if tails is not None:
                if len(output_files) == len(tails):
                    for i, tail in enumerate(tails):
                        self.plotter.filename = self.plotter.filename + tail
                        dataframe = pd.read_csv(f'{output_files[i]}.csv')
                        self.plotter(dataframe)
                        self.plotter.filename = self.plotter.filename.replace(tail, "")
                else:
                    raise Exception(f"`output_files` {len(output_files)} != `tails` {len(tails)}")
            elif input_path:
                dataframe = pd.read_csv(input_path)
                self.plotter(dataframe)
            elif results:
                self.plotter(results)
            else:
                dataframe = pd.read_csv(f'{output_file}.csv')
                self.plotter(dataframe)
            log.info("Done.")

