package chipyard.config

import freechips.rocketchip.config.Config

class LUTAccRocketConfig extends Config(
  new lutacc.WithLUTAcc ++
  new chipyard.config.AbstractConfig)
