import json
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

class MetricsStore():
    def __init__(self, config):
        self.sample_results = {}
        self.event_results = {}
        self.config = config
        pass

    # def mask2tsv(mask, fs, loc, dateTime):
    #     """ Convert a mask of 1/0 (seizure/non-seizure) samples to a list of seizure events.
    #     The list is saved as a .tsv file in the desired format.
        
    #     Arguments:
    #     ---------
    #         - mask:     list of 0/1 for each sample
    #         - fs:       sampling rate
    #         - loc:      destination path of the .tsv file
    #         - dateTime: dateTime to add to the .tsv file (check original .edf file)
    #     """
        
    #     annotator = Annotations.loadMask(mask, fs)
    #     for annotation in annotator.events:
    #         annotation["dateTime"] = dateTime
    #     annotator.saveTsv(loc)

    def evaluate_multiple_predictions(
            self, reference, predictions, patients, fs = 200
    ) -> None:
        for i in range(len(patients)):
            self.evaluate_predictions(reference, fs, predictions, fs, patients )
        return None

    def evaluate_predictions(
        self, reference, ref_fs, predictions, pred_fs, patient
    ) -> None:
        """
        Compares two lists of seizure annotations accross a full dataset and stores the performance in 
        self.sample_results and self.event_results

        Parameters:
        reference (list): The mask of seizure events.
        predictions (list): The mask of predictions.
        patient (string): The name of the subject.
        """

        FS = 1
        #TODO: check if these two sample_results and event_results should be self.sample_results and self.event_results
        sample_results = dict()
        event_results = dict()
        if patient not in self.sample_results.keys():
            self.sample_results[patient] = Result()
            self.event_results[patient] = Result()
            
        # Load annotations
        ref = Annotation(reference, ref_fs)
        pred = Annotation(predictions, pred_fs)

        # Resample annotations
        ref = Annotation(ref.events, FS, int(len(reference)*FS/ref_fs))
        pred = Annotation(pred.events, FS, int(len(predictions)*FS/ref_fs))

        # Compute evaluation
        sample_score = scoring.SampleScoring(ref, pred)
        event_score = scoring.EventScoring(ref, pred)
        # print(self.event_results[patient])

        # Store results
        self.sample_results[patient] = self.sample_results[patient] + Result(sample_score)
        self.event_results[patient] = self.event_results[patient] + Result(event_score)

        return None

    def _compute_scores(self, avg_per_subject = True):

        # Compute scores
        for patient in self.sample_results.keys():
            self.sample_results[patient].computeScores()
            self.event_results[patient].computeScores()

        aggregated_sample_results = dict()
        aggregated_event_results = dict()
        if avg_per_subject:
            for result_builder, aggregated_result in zip(
                (self.sample_results, self.event_results),
                (aggregated_sample_results, aggregated_event_results),
            ):
                for metric in ["sensitivity", "precision", "f1", "fpRate"]:

                    aggregated_result[metric] = np.nanmean(
                        [getattr(x, metric) for x in result_builder.values()]
                    )
                    aggregated_result[f"{metric}_std"] = np.nanstd(
                        [getattr(x, metric) for x in result_builder.values()]
                    )
        else:
            for result_builder, aggregated_result in zip(
                (self.sample_results, self.event_results),
                (aggregated_sample_results, aggregated_event_results),
            ):
                result_builder["cumulated"] = Result()
                for result in result_builder.values():
                    result_builder["cumulated"] += result
                result_builder["cumulated"].computeScores()
                for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                    aggregated_result[metric] = getattr(result_builder["cumulated"], metric)

        output = {
            "sample_results": aggregated_sample_results,
            "event_results": aggregated_event_results,
        }
        return output

    def store_scores(self, sampleOutDir: Path= None, eventOutDir: Path= None):
        res = {"Sample results": self.sample_results,
                "Event results": self.event_results}
        
        name = self.config.model.save_dir.split('/')[-1]


        # with open(sampleOutDir+"/sampleResults_"+name+".json", "w") as file:
        #     json.dump(self.sample_results, file, indent=2, sort_keys=False)
        #
        # with open(eventOutDir+"/eventResults_"+name+".json", "w") as file:
        #     json.dump(self.self.event_results, file, indent=2, sort_keys=False)

        return res

    def store_metrics(self, outDir: Path= None):
        output = self._compute_scores()

        name = self.config.model.save_dir.split('/')[-1]

        # with open(outDir+"/metrics_"+name+".json", "w") as file:
        #     json.dump(output, file, indent=2, sort_keys=False)

        return output


class Result(scoring._Scoring):
    """Helper class built on top of scoring._Scoring that implements the sum
    operator between two scoring objects. The sum corresponds to the
    concatenation of both objects.
    Args:
        scoring (scoring._Scoring): initialized as None (all zeros) or from a
                                    scoring._Scoring object.
    """

    def __init__(self, score: scoring._Scoring = None):
        if score is None:
            self.fs = 0
            self.duration = 0
            self.numSamples = 0
            self.tp = 0
            self.fp = 0
            self.refTrue = 0
        else:
            self.fs = score.ref.fs
            self.duration = len(score.ref.mask) / score.ref.fs
            self.numSamples = score.numSamples
            self.tp = score.tp
            self.fp = score.fp
            self.refTrue = score.refTrue

    def __add__(self, other_result: scoring._Scoring):
        new_result = Result()
        new_result.fs = other_result.fs
        new_result.duration = self.duration + other_result.duration
        new_result.numSamples = self.numSamples + other_result.numSamples
        new_result.tp = self.tp + other_result.tp
        new_result.fp = self.fp + other_result.fp
        new_result.refTrue = self.refTrue + other_result.refTrue

        return new_result

    def __iadd__(self, other_result: scoring._Scoring):
        self.fs = other_result.fs
        self.duration += other_result.duration
        self.numSamples += other_result.numSamples
        self.tp += other_result.tp
        self.fp += other_result.fp
        self.refTrue += other_result.refTrue

        return self

