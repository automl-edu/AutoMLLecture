from genericWrapper4AC.generic_wrapper import AbstractWrapper
from genericWrapper4AC.domain_specific.satwrapper import SatWrapper

class Satenstein_Wrapper(SatWrapper):

    def __init__(self):
        SatWrapper.__init__(self)

    def get_command_line_args(self, runargs, config):
        '''
        @contact:    lindauer@informatik.uni-freiburg.de, fh@informatik.uni-freiburg.de
        Returns the command line call string to execute the target algorithm (here: Satenstein).
        Args:
            runargs: a map of several optional arguments for the execution of the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,                               !!! should be mapped to timeout !!!
                      "runlength" : <runlength cutoff>,                          !!! should be mapped to cutoff  !!!
                      "seed" : <seed>
                    }
            config: a mapping from parameter name to parameter value
        Returns:
            A command call list to execute the target algorithm.
        '''
        solver_binary = "satenstein/ubcsat"

        # Construct the call string to glucose.
        cmd = "%s -alg satenstein" % (solver_binary)

        for name, value in config.items():
            pass # TODO

        #TODO rest of the command

        # remember instance and cmd to verify the result later on
        self._instance = runargs["instance"]
        self._cmd = cmd

        return cmd

if __name__ == "__main__":
    wrapper = Satenstein_Wrapper()
    wrapper.main()
