import abc

class IDataLoader(abc.ABC):

    @abc.abstractmethod
    def loadData(self, path: str):
        """Loads data from the path which was specified in the argument."""
        pass
