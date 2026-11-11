# 新OD删失来源定位

原新OD五条censored轨迹首次缺测都在VBZ（agency3849）tram11的Bucheggplatz→Radiostudio：7月28日cid14703（analytic），29日cid7647（四方法）。不是声明源外的运营商。

现有municipal匹配表把该站称为Brunnenhof（halt_id1525），静态GTFS称Radiostudio（UIC8591307）。两日已测同provider trip/vehicle的seq22→23及25→26夹定该站，中间seq23→24→25、线路、相邻站及计划分钟均唯一一致。原crosswalk缺该别名，源有实测却以stop_identity_unresolved排除。

已直接补一条`STOP_NAME_ALIASES`并用原本地源重join28/29日，输出`city_confirmation_source_completed.json.gz`和同stem audit。新增实测边146/117；原实测23647/15975边的时钟与provider身份不变，25日训练record完全不变。原锁和原数据保留；不调模型，不换OD，不删轨迹。来源补全后的冻结策略回放属于source-completion sensitivity，不能作为新的独立模型确认。
