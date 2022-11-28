
import os


def create_directory(filename):
    """
    Creates a directory for a filename if it doesn't exist already

    :param filename: The path of the filename
    """
    os.makedirs(filename, exist_ok=True)
    #if '/' in filename:
     #   folders = filename.split('/')[:-1]
      #  for pos in range(len(folders)):
       #     if not os.path.exists('/'.join(folders[:pos + 1])):
        #        os.mkdir('/'.join(folders[:pos + 1]))
