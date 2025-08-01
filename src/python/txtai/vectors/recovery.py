"""
Recovery module
"""

import os
import shutil


class Recovery:
    """
    Vector embeddings recovery. This class handles streaming embeddings from a vector checkpoint file.
    """

    def __init__(self, checkpoint, vectorsid, load):
        """
        Creates a Recovery instance.

        Args:
            checkpoint: checkpoint directory
            vectorsid: vectors uid for current configuration
            load: load embeddings method
        """

        self.spool, self.path, self.load = None, None, load

        # Get unique file id
        path = f"{checkpoint}/{vectorsid}"
        if os.path.exists(path):
            # Generate recovery path
            self.path = f"{checkpoint}/recovery"

            # Copy current checkpoint to recovery
            shutil.copyfile(path, self.path)

            # Open file an return
            # pylint: disable=R1732
            self.spool = open(self.path, "rb")

    def __call__(self):
        """
        Reads and returns the next batch of embeddings.

        Returns
            batch of embeddings
        """

        try:
            return self.load(self.spool) if self.spool else None
        except EOFError:
            # End of spool file, cleanup
            self.spool.close()
            os.remove(self.path)

            # Clear parameters
            self.spool, self.path = None, None

            return None
