ALL_EXPERIMENTS = {
    6:"unix", # unix system log, 11.023 lines, 856 signatures
    11: "bgl2", # every 10th line of bgl2 system log, ~470.000 lines, ~450 signatures, replaced timestamps, node ids etc with placeholder
    13: "spirit2", # 770.000 log lines, ~700 signatures
    17: "android",
    18: "apache",
    20: "hadoop",
    23: "healthapp",
    24: "hpc",
    25: "linux",
    26: "mac",
    27: "openstack",
    28: "proxifier",
    30: "ssh",
    33: "zookeeper",
}

EXPERIMENT_LIB_MAPPING = {
    11:"bgl2",
    13:"spirit2",
    17:"android",
    18:"apache",
    20: "hadoop",
    23: "healthapp",
    24: "hpc",
    25: "linux",
    26: "mac",
    27: "openstack",
    28: "proxifier",
    30: "ssh",
    33: "zookeeper",
}

SPLIT_TOKEN ={
    "default":['.', '"' , "'" , ',' , '(', ')', '!', '?', ';', ':', "\\" , '/', '[',']',"=","-",'_' ],
    10:['"'," "],
}
