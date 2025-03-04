import pickle
from medmodels.medrecord import MedRecord
from medmodels.statistic_evaluations.evaluate_compare.evaluate import CohortEvaluator

data_dir = "/home/sysadmin/Martin/medmodels/data/40000/"
medrecord = MedRecord.from_ron(data_dir + "mimic_medrecord.ron")
synthetic_medrecord = MedRecord.from_ron(data_dir + "mimic_medrecord.ron")

# creating CohortEvaluators from the same MedRecord doesn't need much
evaluator_mimic = CohortEvaluator(
    medrecord, name="Original MedRecord", patient_group="patients",
)

# save to pickle
with open(data_dir + "evaluator_mimic.pkl", "wb") as f:
    pickle.dump(evaluator_mimic, f)
