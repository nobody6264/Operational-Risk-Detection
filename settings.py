class MetaRelations(object):
    OPE_RE = ('staff', 'operate', 'address')
    OPE_RE_ = ('address', 'operate_', 'staff')
    DEL_RE = ('address', 'delete', 'system')
    DEL_RE_ = ('system', 'delete_', 'address')
    ADD_RE = ('address', 'add', 'system')
    ADD_RE_ = ('system', 'add_', 'address')
    UPD_RE = ('address', 'update', 'system')
    UPD_RE_ = ('system', 'update_', 'address')
    QUE_RE = ('address', 'query', 'system')
    QUE_RE_ = ('system', 'query_', 'address')
    DOW_RE = ('address', 'download', 'system')
    DOW_RE_ = ('system', 'download_', 'address')


class HyperParam(object):
    NUM_NTYPES = 3
    NUM_ETYPES = 12
    N_INP_PER_NTYPE = {0: 5, 1: 2, 2: 3}
    N_HID = 8
    N_LAYERS = 2
    N_HEADS = 4
    EMBEDDING_DIM = 2
    LR = 0.01
    WEIGHT_DECAY = 0.001
    EPOCHS = 100
    ALPHA = 0.5
    NODE_LABELS = 3
    EDGE_LABELS = 2
    CALC_ENERGY = True
