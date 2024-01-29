
import os
from monai.apps import download_and_extract


def download_oasis():
	root_dir = '.'
	resource = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar"

	compressed_file = os.path.join(root_dir, "neurite-oasis.v1.0.tar")
	data_dir = os.path.join(root_dir, "OASIS")
	if not os.path.exists(data_dir):
		os.mkdir(data_dir)
		download_and_extract(resource, compressed_file, data_dir)


if __name__ == "__main__":
	download_oasis()
