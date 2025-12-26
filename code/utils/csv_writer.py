import csv

class CSVLogger:
    def __init__(self, csv_path):
        """
        Initializes CSV file and writes header
        """
        self.csv_file = open(csv_path, mode="w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["frame", "x", "y", "visible"])

    def write(self, frame_id, x, y, visible):
        """
        Write one frame annotation
        """
        self.writer.writerow([frame_id, x, y, visible])

    def close(self):
        """
        Close CSV file safely
        """
        self.csv_file.close()
