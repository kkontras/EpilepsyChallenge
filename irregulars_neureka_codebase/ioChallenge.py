from utils.config import process_config, setup_logger, process_config_default
from datetime import datetime
import csv
from pathlib import Path
from ioDataloader import IODataloader
from IOLoader.recording_reader import Recording_Reader
from ioValidate import main_validate
from pathlib import Path
import argparse
import shutil
shutil._USE_CP_SENDFILE = False

class IOChallenge:
    def __init__(self, pathIn: Path, config):
        self.config = config
        self.recording = Recording_Reader.loadData(pathIn)
        self.recording.preprocessData()
        self.dataloader = IODataloader(self.recording, self.config).loader

    def predict(self):
        return main_validate(self.config, self.dataloader)

    def saveEvents(self, evs: list, pathOut: Path):
        dateTime = self.recording.getTime()
        if not dateTime:
            dateTime = datetime.fromtimestamp(0)
        with open(pathOut, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["onset","duration","eventType","confidence","channels","dateTime", "recordingDuration"])
            if len(evs)> 0:
                for ev in evs:
                    writer.writerow([float(ev[0]), float(ev[1]-ev[0]), "sz", "n/a", "n/a", dateTime, float(self.recording.getDuration())])
            else:
                writer.writerow([0.0, float(self.recording.getDuration()), "bckg", "n/a", "n/a", dateTime, float(self.recording.getDuration())])
        return
    
def main(pathIn, pathOut):
    setup_logger()
    config = process_config_default(Path("irregulars_neureka_codebase/configs/neureka.json"), Path("irregulars_neureka_codebase/configs/default_config_tuhsz2.json"))
    ioChallenge = IOChallenge(pathIn, config)
    prds = ioChallenge.predict()
    ioChallenge.saveEvents(prds, pathOut)

# if __name__=="__main__":
#     pathIn = "/users/sista/mvanmarc/Documents/Doctoraat/SUBJ-1a-006_r10.edf"
#     pathOut = "/users/sista/mvanmarc/Documents/Doctoraat/Python/Challenge16Feb2025/EpilepsyChallenge/irregulars_neureka_codebase/predictions/test.tsv"
#     main(pathIn, pathOut)


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--pathIn', required = True, help="Path to EDF file")
parser.add_argument('--pathOut', required = True, help="Path to tsv file")
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

print(args)


main(args.pathIn, args.pathOut)
    