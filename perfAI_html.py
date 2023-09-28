import pandas as pd
import sys
from decimal import Decimal
from tqdm import tqdm
from numpy import transpose
import os
import shutil
import json
from functools import reduce
from collections import defaultdict

def intToHex(dataList):
    newDataList = []
    for data in dataList:
        if not data.isnumeric():
            newDataList.append('')
        else:
            newDataList.append(str(hex(int(data))))
    return newDataList

def getDataTypeMap():
    data_type_dict = {
        '0': 'INT8',
        '1': 'FP16',
        '2': 'FP32',
        '3': 'INT16',
        '4': 'INT32',
        '5': 'BFP16',
        '6': 'INT64',
        '7': 'FP20',
        '': 'None',
        '-': 'None'
    }
    return data_type_dict


def getDataSizeMap():
    data_size_dict = {
        '0': 1,
        '1': 2,
        '2': 4,
        '3': 2,
        '4': 4,
        '5': 2,
        '6': 8,
        '': 'None',
        '-': 'None'
    }
    return data_size_dict

def getTiuColumns(): 
    cols = ['Engine Id', 'Core Id', 'Cmd Id', 'Function Type', 'Function Name', 'Alg Cycle',
            'Simulator Cycle', 'Start Cycle', 'End Cycle', 'Alg Ops', 'uArch Ops', 'uArch Rate', 'Bank Conflict Ratio',
            'Initial Cycle Ratio', 'Data Type', 'des_cmd_id_dep',
            'des_res0_n', 'des_res0_c', 'des_res0_h', 'des_res0_w',
            'des_res0_n_str', 'des_res0_c_str', 'des_res0_h_str', 'des_res0_w_str',
            'des_opd0_n', 'des_opd0_c', 'des_opd0_h', 'des_opd0_w',
            'des_opd0_n_str', 'des_opd0_c_str', 'des_opd0_h_str', 'des_opd0_w_str',
            'des_opd1_n','des_opd1_c', 'des_opd1_h', 'des_opd1_w', 'des_opd1_n_str', 'des_opd1_c_str', 'des_opd1_h_str',
            'des_opd1_w_str',
            'des_opd2_n_str', 'des_res0_addr', 'des_res1_addr', 'des_opd0_addr', 'des_opd1_addr', 'des_opd2_addr',
            'des_opd3_addr', 'des_tsk_typ', 'des_tsk_eu_typ', 'des_cmd_short',
            'des_opt_res0_prec', 'des_opt_opd0_prec', 'des_opt_opd1_prec', 'des_opt_opd2_prec', 'des_short_opd0_str',
            'des_opt_opd0_const', 'des_opt_opd1_const', 'des_opt_opd2_const', 'des_opt_opd3_const',
            'des_opt_opd4_const', 'des_opt_opd5_const',
            'des_opt_res_add', 'des_opt_res0_sign', 'des_opt_opd0_sign', 'des_opt_opd1_sign', 'des_opt_opd2_sign',
            'des_opd0_rt_pad', 'des_opd1_x_ins0', 'des_opd0_up_pad', 'des_opd0_lf_pad', 'des_opt_left_tran',
            'des_pad_mode', 'des_opd0_y_ins0', 'des_opd1_y_ins0',
            'des_short_res0_str', 'des_short_opd1_str', 'des_sym_range', 'des_opt_rq', 'des_op_code',
            'des_opt_kernel_rotate', 'des_res_op_x_str', 'des_res_op_y_str', 'des_opd0_x_ins0',
            'des_tsk_opd_num', 'des_opd0_dn_pad', 'des_intr_en', 'des_opt_relu', 'des_pwr_step', 'Msg Id', 'Sd\Wt Count']
    return cols

def getGdmaColumns():
    cols = ['Engine Id', 'Core Id', 'Cmd Id', 'Function Type', 'Function Name', 'DMA data size(B)', 'Start Cycle', 'End Cycle',
            'Simulator Cycle', 'Bandwidth', 'Direction', 'AvgBurstLength', 'Data Type', 'Non32ByteRatio',
            'MaskWriteRatio', 'cmd_special_function', 'src_start_addr', 'dst_start_addr',
            'src_shape','dst_shape', 'index_shape', #(size, datatype, stride)
            'nchw_copy', 'stride_enable', 'src_data_format', 'cmd_type',
            'mask_start_addr_h8', 'mask_start_addr_l32', 'mask_data_format', 'localmem_mask_h32', 'localmem_mask_l32',
            'fill_constant_en', 'constant_value', 'index', 'cmd_short', 'cmd_id_dep', 'intr_en', 'Msg Id', 'Sd\Wt Count']
    return cols

def getCdmaColumns():
    cols = ['Engine Id', 'Core Id', 'Cmd Id',  'Function Name', 'DMA data size(B)', 'Start Cycle', 'End Cycle',
            'Simulator Cycle', 'Bandwidth', 'Direction', 'AvgBurstLength', 'Data Type', 'Non32ByteRatio',
            'MaskWriteRatio', 'cmd_special_function', 'src_start_addr', 'dst_start_addr',
            'src_nsize', 'src_csize', 'src_hsize', 'src_wsize',
            'dst_nsize', 'dst_csize', 'dst_hsize', 'dst_wsize',
            'src_nstride', 'src_cstride', 'src_hstride',
            'dst_nstride', 'dst_cstride', 'dst_hstride',
            'nchw_copy', 'stride_enable', 'src_data_format', 'cmd_type',
            'fill_constant_en', 'constant_value', 'index', 'intr_en', 'Msg Id', 'Sd\Wt Count']
    return cols

def RegInfo(tiuRegFile, gdmaRegFile, cdmaRegFile, simulatorFile, coreNum, chipArchArgs, linecount):
    def get_simulator_total_cycle(simulatorFile):
        simulatorTotalCycle = 0
        if os.path.exists(simulatorFile):
            with open(simulatorFile) as f:
                rows = f.readlines()
                for row in rows:
                    if ':' in row:
                        simulatorTotalCycle = int(row.split(': ')[1])
        return simulatorTotalCycle

    simulatorTotalCycle = get_simulator_total_cycle(simulatorFile)

    def getTotalInstCols(tiuCols, dmaCols):
        tiuColSet = set(tiuCols)
        for col in dmaCols:
            if col not in tiuColSet:
                tiuCols.append(col)
        return tiuCols

    def getDmaFunctionName(regDict):
        dmaFunctionNameDict = {
            (0, 0):'DMA_tensor', (0, 1):'NC trans', (0, 2):'collect', (0, 3):'broadcast', (0, 4):'distribute', (0, 5):'lmem 4 bank copy', (0, 6):'lmem 4 bank broadcast',
            (1, 0):'DMA_matrix', (1, 1):'matrix transpose',
            (2, 0):'DMA_masked_select', (2, 1):'ncw mode',
            (3, 0):'DMA_general', (3, 1):'broadcast',
            (4, 0):'cw transpose',
            (5, 0):'DMA_nonzero',
            (6, 0):'chain end', (6, 1):'nop', (6, 2):'sys_tr_wr', (6, 3):'sys_send', (6, 4):'sys_wait',
            (7, 0):'DMA_gather',
            (8, 0):'DMA_scatter',
            (9, 0):'w reverse', (9, 1):'h reverse', (9, 2):'c reverse', (9, 3):'n reverse',
            (10, 0):'non-random-access', (10, 1):'random-access',
            (11, 0):'non-random-access', (11, 1):'random-access'
        }
        dmaFunctionName = ''
        dmaCmdType, dmaSpecialFuncName = int(regDict['cmd_type']), int(regDict['cmd_special_function'])
        if (dmaCmdType, dmaSpecialFuncName) in dmaFunctionNameDict.keys() and \
              dmaFunctionNameDict[(dmaCmdType, dmaSpecialFuncName)] != regDict['Function Type']:
            dmaFunctionName = dmaFunctionNameDict[(dmaCmdType, dmaSpecialFuncName)]
        else:
            direction = ''
            if 'DDR' not in regDict['Direction']:
                direction = 'Mv'
            else:
                dirs = regDict['Direction'].split('->')
                if dirs[0] == 'DDR':
                    direction = 'Ld'
                elif dirs[1] == 'DDR':
                    direction = 'St'
            dmaFunctionName += regDict['Function Type'].split('_')[1] + direction
        return dmaFunctionName

    def get_start_addr(addr_h8, addr_l32):
        addr_h8 = str(hex(int(addr_h8)))[2:]
        addr_l32 = str(hex(int(addr_l32)))[2:]
        while(len(addr_h8) < 2):
            addr_h8 = '0' + addr_h8
        while(len(addr_l32) < 8):
            addr_l32 = '0' + addr_l32
        return '0x' + addr_h8 + addr_l32

    # set total summary data
    TiuWorkingRatioList, SimTotalCycleList, SimTiuCycleList, TotalAlgCycleList, TotalAlgOpsList, TotaluArchOpsList, ParallelismList, uArchRateList = [], [], [], [], [], [], [], []
    SimGdmaCycleList, GdmaDdrTotalDatasizeList, GdmaL2TotalDatasizeList, GdmaDdrAvgBandwidthList, GdmaL2AvgBandwidthList, gdmaDdrAvgBurstLengthList = [], [], [], [], [], []
    gdmaTotalBurstLength, gdmaTotalXactCnt, GdmaDdrTotalCycle, GdmaL2TotalCycle, gdmaWorkingCycle = 0, 0, 0, 0, 0
    SimSdmaCycleList, SdmaL2TotalDatasizeList, SdmaDdrAvgBandwidthList, SdmaL2AvgBandwidthList, SdmaDdrTotalDatasizeList, sdmaDdrAvgBurstLengthList = [], [], [], [], [], []
    sdmaTotalBurstLength, sdmaTotalXactCnt, SdmaDdrTotalCycle, SdmaL2TotalCycle, sdmaWorkingCycle = 0, 0, 0, 0, 0
    SimCdmaCycleList, CdmaDdrTotalDatasizeList, CdmaL2TotalDatasizeList, CdmaDdrAvgBandwidthList, CdmaL2AvgBandwidthList, cdmaDdrAvgBurstLengthList = [], [], [], [], [], []
    cdmaTotalBurstLength, cdmaTotalXactCnt, CdmaDdrTotalCycle, CdmaL2TotalCycle, cdmaWorkingCycle = 0, 0, 0, 0, 0
    totalInstRegList = []
    actualCoreNum = 0
    for coreId in range(int(coreNum)):
        curTiuRegFile, curGdmaRegFile, curCdmaRegFile = tiuRegFile + '_' + str(coreId) + '.txt', gdmaRegFile + '_' + str(coreId) + '.txt', cdmaRegFile + '_' + str(coreId) + '.txt'
        if (os.path.exists(curTiuRegFile) and os.path.getsize(curTiuRegFile)) or (os.path.exists(curGdmaRegFile) and os.path.getsize(curGdmaRegFile)):
            actualCoreNum += 1
    summaryDf = pd.DataFrame()
    alldata = []
    for coreId in range(actualCoreNum):
        tiuRegList, gdmaRegList, sdmaRegList, cdmaRegList = [], [], [], []
        simulatorTiuCycle, simulatorGdmaCycle, simulatorSdmaCycle, simulatorCdmaCycle = 0, 0, 0, 0
        algTotalCycle = 0
        algTotalOps = 0
        uArchTotalOps = 0
        tiuWaitMsgTotalTime = 0
        gdmaWaitMsgTotalTime = 0
        sdmaWaitMsgTotalTime = 0
        cdmaWaitMsgTotalTime = 0
        gdmaDdrTotalDataSize, gdmaL2TotalDataSize, sdmaDdrTotalDataSize, sdmaL2TotalDataSize, cdmaDdrTotalDataSize, cdmaL2TotalDataSize = 0, 0, 0, 0, 0, 0
        gdmaDdrCycle, gdmaL2Cycle, sdmaDdrCycle, sdmaL2Cycle, cdmaDdrCycle, cdmaL2Cycle = 0, 0, 0, 0, 0, 0
        gdmaDdrBurstLength, sdmaDdrBurstLength, gdmaDdrXactCnt, sdmaDdrXactCnt, cdmaDdrBurstLength, cdmaDdrXactCnt = 0, 0, 0, 0, 0, 0

        # TIU
        curTiuRegFile = tiuRegFile + '_' + str(coreId) + '.txt'
        if os.path.exists(curTiuRegFile) and os.path.getsize(curTiuRegFile) != 0:
            with open(curTiuRegFile) as f:
                rows = f.readlines()[linecount:]
                fieldSet = set()
                for row in rows:
                    if "\t" in row:
                        attr = row.split(': ')[0][1:]
                        fieldSet.add(attr)
                tiuCols = getTiuColumns()
                fieldList = list(fieldSet) if len(fieldSet) >= len(tiuCols) else tiuCols
                tiuRegDict = dict.fromkeys(fieldList, '')
                idx = 0
                for row in rows:
                    if "__TIU_REG_INFO__" in row:
                        if idx != 0:
                            tiuRegList.append(tiuRegDict)
                            tiuRegDict = dict.fromkeys(fieldList, '')
                    elif "\t" not in row:
                        tiuRegDict['Function Type'] = row[:-2]
                    else:
                        fields = row.split(': ')
                        attr = fields[0][1:]
                        val = fields[1][:-1]
                        tiuRegDict[attr] = val
                    idx += 1
                tiuRegList.append(tiuRegDict)
        data_type_dict = getDataTypeMap()

        for i in range(len(tiuRegList)):
            regDict = tiuRegList[i]
            if regDict['Simulator Cycle'].isnumeric():
                simulatorTiuCycle += int(regDict['Simulator Cycle'])
            if regDict['Alg Cycle'].isnumeric():
                algTotalCycle += int(regDict['Alg Cycle'])
            if regDict['Alg Ops'].isnumeric():
                algTotalOps += int(regDict['Alg Ops'])
            if regDict['uArch Ops'].isnumeric():
                uArchTotalOps += int(regDict['uArch Ops'])
            if data_type_dict[regDict['des_opt_opd0_prec']].isnumeric():
                regDict['Data Type'] = data_type_dict[regDict['des_opt_res0_prec']] + \
                                        ' -> ' + data_type_dict[regDict['des_opt_res0_prec']]
            else:
                regDict['Data Type'] = data_type_dict[regDict['des_opt_res0_prec']] + \
                                        ' -> ' + data_type_dict[regDict['des_opt_res0_prec']]
            totalInstRegList.append(regDict)
            if int(regDict['des_tsk_typ']) == 15 and int(regDict['des_tsk_eu_typ']) == 9:
                tiuWaitMsgTotalTime += int(regDict['Simulator Cycle'])

        #GDMA
        curGdmaRegFile = gdmaRegFile + '_' + str(coreId) + '.txt'
        if os.path.exists(curGdmaRegFile) and os.path.getsize(curGdmaRegFile) != 0:
            with open(curGdmaRegFile) as f:
                rows = f.readlines()[linecount:]
                fieldSet = set()
                for row in rows:
                    if "\t" in row:
                        attr = row.split(': ')[0][1:]
                        fieldSet.add(attr)
                dmaCols = getGdmaColumns()
                fieldList = list(fieldSet) if len(fieldSet) >= len(dmaCols) else dmaCols
                gdmaRegDict = dict.fromkeys(fieldList, '')
                idx = 0
                for row in rows:
                    if "__TDMA_REG_INFO__" in row:
                        if idx != 0:
                            if gdmaRegDict['Engine Id'] == '1':
                                gdmaRegList.append(gdmaRegDict)
                            elif gdmaRegDict['Engine Id'] == '3':
                                sdmaRegList.append(gdmaRegDict)
                        gdmaRegDict = dict.fromkeys(fieldList, '')
                    else:
                        fields = row.split(': ')
                        attr = fields[0][1:]
                        val = fields[1][:-1]
                        gdmaRegDict[attr] = val
                    idx += 1
                if gdmaRegDict['Engine Id'] == '1':
                    gdmaRegList.append(gdmaRegDict)
                elif gdmaRegDict['Engine Id'] == '3':
                    sdmaRegList.append(gdmaRegDict)
                else:
                    print('Not Support Engine Id')
                    exit(-1)

        frequency = int(chipArchArgs['Frequency(MHz)'])

        for i in range(len(gdmaRegList)):
            regDict = gdmaRegList[i]
            if regDict['Simulator Cycle'].isnumeric():
                simulatorGdmaCycle += int(regDict['Simulator Cycle'])
            if CHIP_ARCH in ('bm1684x', 'cv186x', 'A2', 'mars3'):
                regDict['src_start_addr'] = get_start_addr(regDict['src_start_addr_h8'], regDict['src_start_addr_l32'])
                regDict['dst_start_addr'] = get_start_addr(regDict['dst_start_addr_h8'], regDict['dst_start_addr_l32'])

            if 'DDR' in regDict['Direction'] and regDict['DMA data size(B)'].isnumeric():
                    if regDict['Direction'].count('DDR') == 2:
                        gdmaDdrTotalDataSize += int(regDict['gmem_xfer_bytes(B)'])
                        regDict['Bandwidth'] = Decimal(int(regDict['gmem_xfer_bytes(B)']) / int(regDict['Simulator Cycle'])).quantize(Decimal("0.00"))
                    else:
                        gdmaDdrTotalDataSize += int(regDict['DMA data size(B)'])
                    gdmaDdrCycle += float(regDict['Simulator Cycle'])
                    gdmaDdrBurstLength += int(regDict['gmem_bl_sum'])
                    gdmaDdrXactCnt += int(regDict['gmem_xact_cnt'])
            elif 'L2' in regDict['Direction'] and regDict['DMA data size(B)'].isnumeric():
                    gdmaL2TotalDataSize += int(regDict['DMA data size(B)'])
                    gdmaL2Cycle += float(regDict['Simulator Cycle'])
            if int(regDict['cmd_type']) == 6 and int(regDict['cmd_special_function']) == 4:
                gdmaWaitMsgTotalTime += int(regDict['Simulator Cycle'])
            if int(regDict['gmem_xact_cnt']) > 0:
                regDict['AvgBurstLength'] = Decimal(
                    int(regDict['gmem_bl_sum']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['Non32ByteRatio'] = Decimal(
                    int(regDict['gmem_n32Ba_sa_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['MaskWriteRatio'] = Decimal(
                    int(regDict['gmem_msk_wr_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
            else:
                regDict['AvgBurstLength'] = 0
                regDict['Non32ByteRatio'] = 0
                regDict['MaskWriteRatio'] = 0
            totalInstRegList.append(regDict)
        for i in range(len(sdmaRegList)):
            regDict = sdmaRegList[i]
            if regDict['Simulator Cycle'].isnumeric():
                simulatorSdmaCycle += int(regDict['Simulator Cycle'])
            if CHIP_ARCH in ('bm1684x', 'cv186x', 'A2', 'mars3'):
                    regDict['src_start_addr'] = get_start_addr(regDict['src_start_addr_h8'], regDict['src_start_addr_l32'])
                    regDict['dst_start_addr'] = get_start_addr(regDict['dst_start_addr_h8'], regDict['dst_start_addr_l32'])
            if 'DDR' in regDict['Direction']:
                    if regDict['DMA data size(B)'].isnumeric():
                        if regDict['Direction'].count('DDR') == 2:
                            sdmaDdrTotalDataSize += int(regDict['gmem_xfer_bytes(B)'])
                            regDict['Bandwidth'] = Decimal(int(regDict['gmem_xfer_bytes(B)']) / int(regDict['Simulator Cycle'])).quantize(Decimal("0.00"))
                        else:
                            sdmaDdrTotalDataSize += int(regDict['DMA data size(B)'])
                        sdmaDdrCycle += float(regDict['Simulator Cycle'])
                        sdmaDdrBurstLength += int(regDict['gmem_bl_sum'])
                        sdmaDdrXactCnt += int(regDict['gmem_xact_cnt'])
            elif 'L2' in regDict['Direction']:
                if regDict['DMA data size(B)'].isnumeric():
                    sdmaL2TotalDataSize += int(regDict['DMA data size(B)'])
                    sdmaL2Cycle += float(regDict['Simulator Cycle'])
            if int(regDict['cmd_type']) == 6 and int(regDict['cmd_special_function']) == 4:
                sdmaWaitMsgTotalTime += int(regDict['Simulator Cycle'])
            if int(regDict['gmem_xact_cnt']) > 0:
                regDict['AvgBurstLength'] = Decimal(
                    int(regDict['gmem_bl_sum']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['Non32ByteRatio'] = Decimal(
                    int(regDict['gmem_n32Ba_sa_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['MaskWriteRatio'] = Decimal(
                    int(regDict['gmem_msk_wr_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
            else:
                regDict['AvgBurstLength'] = 0
                regDict['Non32ByteRatio'] = 0
                regDict['MaskWriteRatio'] = 0
            totalInstRegList.append(regDict)

        # CDMA
        curCdmaRegFile = cdmaRegFile + '_' + str(coreId) + '.txt'
        if os.path.exists(curCdmaRegFile) and os.path.getsize(curCdmaRegFile) != 0:
            with open(curCdmaRegFile) as f:
                rows = f.readlines()[linecount:]
                fieldSet = set()
                for row in rows:
                    if "\t" in row:
                        attr = row.split(': ')[0][1:]
                        fieldSet.add(attr)
                dmaCols = getCdmaColumns()
                fieldList = list(fieldSet) if len(fieldSet) >= len(dmaCols) else dmaCols
                cdmaRegDict = dict.fromkeys(fieldList, '')
                idx = 0
                for row in rows:
                    if "__CDMA_REG_INFO__" in row:
                        if idx != 0:
                            cdmaRegList.append(cdmaRegDict)
                        gdmaRegDict = dict.fromkeys(fieldList, '')
                    else:
                        fields = row.split(': ')
                        attr = fields[0][1:]
                        val = fields[1][:-1]
                        cdmaRegDict[attr] = val
                    idx += 1
                cdmaRegList.append(cdmaRegDict)

        for i in range(len(cdmaRegList)):
            regDict = cdmaRegList[i]
            if regDict['Simulator Cycle'].isnumeric():
                simulatorCdmaCycle += int(regDict['Simulator Cycle'])
            if 'DDR' in regDict['Direction']:
                if regDict['DMA data size(B)'].isnumeric():
                    if regDict['Direction'].count('DDR') == 2:
                        cdmaDdrTotalDataSize += int(regDict['gmem_xfer_bytes(B)'])
                        regDict['Bandwidth'] = Decimal(int(regDict['gmem_xfer_bytes(B)']) / int(regDict['Simulator Cycle'])).quantize(Decimal("0.00"))
                    else:
                        cdmaDdrTotalDataSize += int(regDict['DMA data size(B)'])
                    cdmaDdrCycle += float(regDict['Simulator Cycle'])
                    cdmaDdrBurstLength += int(regDict['gmem_bl_sum'])
                    cdmaDdrXactCnt += int(regDict['gmem_xact_cnt'])
            elif 'L2' in regDict['Direction']:
                if regDict['DMA data size(B)'].isnumeric():
                    cdmaL2TotalDataSize += int(regDict['DMA data size(B)'])
                    cdmaL2Cycle += float(regDict['Simulator Cycle'])
            if int(regDict['cmd_type']) == 7 and (int(regDict['cmd_special_function']) == 4 or int(regDict['cmd_special_function']) == 6):
                cdmaWaitMsgTotalTime += int(regDict['Simulator Cycle'])
            if int(regDict['gmem_xact_cnt']) > 0:
                regDict['AvgBurstLength'] = Decimal(
                    int(regDict['gmem_bl_sum']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['Non32ByteRatio'] = Decimal(
                    int(regDict['gmem_n32Ba_sa_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
                regDict['MaskWriteRatio'] = Decimal(
                    int(regDict['gmem_msk_wr_cnt']) / int(regDict['gmem_xact_cnt'])).quantize(Decimal("0.00"))
            else:
                regDict['AvgBurstLength'] = 0
                regDict['Non32ByteRatio'] = 0
                regDict['MaskWriteRatio'] = 0
            totalInstRegList.append(regDict)

        tiuDf = pd.DataFrame(tiuRegList)
        tiuCols = getTiuColumns()
        newTiuCols = []
        for tiuCol in tiuCols:
            if tiuCol in tiuDf.columns.values.tolist():
                newTiuCols.append(tiuCol)
        tiuCols = newTiuCols
        tiuDf = tiuDf[tiuCols]
        for col in tiuCols:
            if 'addr' in col or 'mask' in col:
                tiuDf[col] = intToHex(tiuDf[col].values)
        gdmaDf = pd.DataFrame(gdmaRegList)
        sdmaDf = pd.DataFrame(sdmaRegList)
        cdmaDf = pd.DataFrame(cdmaRegList)
        dmaCols = getGdmaColumns()
        cdmaCols = getCdmaColumns()
        if len(gdmaDf) > 0:
            gdmaDf = gdmaDf[dmaCols]
        if len(sdmaDf) > 0:
            sdmaDf = sdmaDf[dmaCols]
        if len(cdmaDf) > 0:
            # dmaCols.remove('Function Type')
            cdmaDf = cdmaDf[cdmaCols]
        for col in dmaCols:
            if CHIP_ARCH == 'sg2260':
                if ('addr' in col or 'mask' in col):
                    if len(gdmaDf) != 0:
                        gdmaDf[col] = intToHex(gdmaDf[col].values)
                    if len(sdmaDf) != 0:
                        sdmaDf[col] = intToHex(sdmaDf[col].values)
            elif CHIP_ARCH in ('mars3', 'cv186x', 'A2'):
                if ('addr' in col or 'mask' in col) and ('start' not in col):
                    gdmaDf[col] = intToHex(gdmaDf[col].values)
                    if len(sdmaDf) != 0:
                        sdmaDf[col] = intToHex(sdmaDf[col].values)
        for col in cdmaCols:
            if CHIP_ARCH == 'sg2260':
                if ('addr' in col or 'mask' in col):
                    if len(cdmaDf) != 0:
                        cdmaDf[col] = intToHex(cdmaDf[col].values)
            elif CHIP_ARCH in ('mars3', 'cv186x', 'A2'):
                if ('addr' in col or 'mask' in col) and ('start' not in col):
                    if len(cdmaDf) != 0:
                        cdmaDf[col] = intToHex(cdmaDf[col].values)

        SimTotalCycleList.append(simulatorTotalCycle)
        SimTiuCycleList.append(simulatorTiuCycle)
        TiuWorkingRatioList.append('0.00%' if simulatorTotalCycle == 0 else
                                       str(Decimal((simulatorTiuCycle * 100 / simulatorTotalCycle)).quantize(
                                           Decimal("0.00"))) + '%')
        if CHIP_ARCH in ("sg2260", "A2"):
            ParallelismList.append('0.00%' if simulatorTotalCycle == 0 else
                                    str(Decimal(((simulatorTiuCycle + simulatorGdmaCycle + simulatorSdmaCycle) * 100 / simulatorTotalCycle)).quantize(
                                        Decimal("0.00"))) + '%')
        elif CHIP_ARCH in ("bm1684x", "mars3", "cv186x"):
            ParallelismList.append('0.00%' if simulatorTotalCycle == 0 else
                                    str(Decimal(((simulatorTiuCycle + simulatorGdmaCycle) * 100 / simulatorTotalCycle)).quantize(
                                        Decimal("0.00"))) + '%')
        else:
            print("Not Support CHIP_ARCH")
            assert(0)

        TotalAlgCycleList.append(algTotalCycle)
        TotalAlgOpsList.append(algTotalOps)
        TotaluArchOpsList.append(uArchTotalOps)
        uArchRateList.append('0.00%' if uArchTotalOps == 0 else str(
                        Decimal((algTotalOps * 100 / uArchTotalOps)).quantize(Decimal("0.00"))) + '%')

        SimGdmaCycleList.append(simulatorGdmaCycle)
        GdmaDdrTotalDatasizeList.append(gdmaDdrTotalDataSize)
        GdmaL2TotalDatasizeList.append(gdmaL2TotalDataSize)
        if gdmaDdrCycle > 0:
            gdmaDdrTotalBandWidth = str(Decimal((gdmaDdrTotalDataSize / gdmaDdrCycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            gdmaDdrTotalBandWidth = 0
        if gdmaL2Cycle > 0:
            gdmaL2TotalBandWidth = str(Decimal((gdmaL2TotalDataSize / gdmaL2Cycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            gdmaL2TotalBandWidth = 0
        GdmaDdrTotalCycle += gdmaDdrCycle
        GdmaL2TotalCycle += gdmaL2Cycle
        GdmaDdrAvgBandwidthList.append(gdmaDdrTotalBandWidth)
        GdmaL2AvgBandwidthList.append(gdmaL2TotalBandWidth)
        gdmaDdrAvgBurstLength = 0 if gdmaDdrXactCnt == 0 else Decimal((gdmaDdrBurstLength / gdmaDdrXactCnt)).quantize(
            Decimal("0.00"))
        gdmaDdrAvgBurstLengthList.append(gdmaDdrAvgBurstLength)
        gdmaTotalBurstLength += gdmaDdrBurstLength
        gdmaTotalXactCnt += gdmaDdrXactCnt

        SimSdmaCycleList.append(simulatorSdmaCycle)
        SdmaDdrTotalDatasizeList.append(sdmaDdrTotalDataSize)
        SdmaL2TotalDatasizeList.append(sdmaL2TotalDataSize)
        sdmaDdrAvgBurstLength = 0 if sdmaDdrXactCnt == 0 else Decimal((sdmaDdrBurstLength / sdmaDdrXactCnt)).quantize(
            Decimal("0.00"))
        if sdmaDdrCycle > 0:
            sdmaDdrTotalBandWidth = str(Decimal((sdmaDdrTotalDataSize / sdmaDdrCycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            sdmaDdrTotalBandWidth = 0
        if sdmaL2Cycle > 0:
            sdmaL2TotalBandWidth = str(Decimal((sdmaL2TotalDataSize / sdmaL2Cycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            sdmaL2TotalBandWidth = 0
        SdmaDdrTotalCycle += sdmaDdrCycle
        SdmaL2TotalCycle += sdmaL2Cycle
        SdmaDdrAvgBandwidthList.append(sdmaDdrTotalBandWidth)
        SdmaL2AvgBandwidthList.append(sdmaL2TotalBandWidth)
        sdmaDdrAvgBurstLengthList.append(sdmaDdrAvgBurstLength)
        sdmaTotalBurstLength += sdmaDdrBurstLength
        sdmaTotalXactCnt += sdmaDdrXactCnt

        SimCdmaCycleList.append(simulatorCdmaCycle)
        CdmaDdrTotalDatasizeList.append(cdmaDdrTotalDataSize)
        CdmaL2TotalDatasizeList.append(cdmaL2TotalDataSize)
        if cdmaDdrCycle > 0:
            cdmaDdrTotalBandWidth = str(Decimal((cdmaDdrTotalDataSize / cdmaDdrCycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            cdmaDdrTotalBandWidth = 0
        if cdmaL2Cycle > 0:
            cdmaL2TotalBandWidth = str(Decimal((cdmaL2TotalDataSize / cdmaL2Cycle * frequency / 1000)).quantize(Decimal("0.00")))
        else:
            cdmaL2TotalBandWidth = 0
        CdmaDdrTotalCycle += cdmaDdrCycle
        CdmaL2TotalCycle += cdmaL2Cycle
        CdmaDdrAvgBandwidthList.append(cdmaDdrTotalBandWidth)
        CdmaL2AvgBandwidthList.append(cdmaL2TotalBandWidth)
        cdmaDdrAvgBurstLength = 0 if cdmaDdrXactCnt == 0 else Decimal((cdmaDdrBurstLength / cdmaDdrXactCnt)).quantize(
            Decimal("0.00"))
        cdmaDdrAvgBurstLengthList.append(cdmaDdrAvgBurstLength)
        cdmaTotalBurstLength += cdmaDdrBurstLength
        cdmaTotalXactCnt += cdmaDdrXactCnt

        alldata.append([tiuDf, gdmaDf, sdmaDf, cdmaDf])

    TiuWorkingRatioList.append('0.00%' if sum(SimTotalCycleList) == 0 else
                                str((Decimal((sum(SimTiuCycleList) / sum(SimTotalCycleList)) * 100)).quantize(
                                    Decimal("0.00"))) + '%')
    if CHIP_ARCH in ("sg2260", "A2"):
        ParallelismList.append('0.00%' if sum(SimTotalCycleList) == 0 else
                                str((Decimal(((max(SimTiuCycleList) + max(SimGdmaCycleList) + max(SimSdmaCycleList)) / max(SimTotalCycleList)) * 100)).quantize(
                                    Decimal("0.00"))) + '%')
    elif CHIP_ARCH in ("bm1684x", "mars3", "cv186x"):
        ParallelismList.append('0.00%' if sum(SimTotalCycleList) == 0 else
                                str((Decimal(((max(SimTiuCycleList) + max(SimGdmaCycleList)) / max(SimTotalCycleList)) * 100)).quantize(
                                    Decimal("0.00"))) + '%')
    uArchRateList.append('0.00%' if sum(TotaluArchOpsList) == 0 else
                        str((Decimal((sum(TotalAlgOpsList) / sum(TotaluArchOpsList)) * 100)).quantize(
                            Decimal("0.00"))) + '%')
    SimTotalCycleList.append(max(SimTotalCycleList))
    SimTiuCycleList.append(max(SimTiuCycleList))
    TotalAlgCycleList.append(max(TotalAlgCycleList))
    TotalAlgOpsList.append(sum(TotalAlgOpsList))
    TotaluArchOpsList.append(sum(TotaluArchOpsList))
    SimGdmaCycleList.append(max(SimGdmaCycleList))
    GdmaDdrTotalDatasizeList.append(sum(GdmaDdrTotalDatasizeList))
    GdmaL2TotalDatasizeList.append(sum(GdmaL2TotalDatasizeList))
    gdmaDdrAvgBurstLengthList.append('0.00' if gdmaTotalXactCnt == 0 else
                                        str((Decimal(gdmaTotalBurstLength / gdmaTotalXactCnt)).quantize(
                                            Decimal("0.00"))) )
    if GdmaDdrTotalCycle == 0:
        GdmaDdrAvgBandwidthList.append(0)
    else:
        GdmaDdrAvgBandwidthList.append(
            str((Decimal((GdmaDdrTotalDatasizeList[-1] / GdmaDdrTotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    if GdmaL2TotalCycle == 0:
        GdmaL2AvgBandwidthList.append(0)
    else:
        GdmaL2AvgBandwidthList.append(
            str((Decimal((GdmaL2TotalDatasizeList[-1] / GdmaL2TotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    #sdma
    SimSdmaCycleList.append(max(SimSdmaCycleList))
    SdmaDdrTotalDatasizeList.append(sum(SdmaDdrTotalDatasizeList))
    SdmaL2TotalDatasizeList.append(sum(SdmaL2TotalDatasizeList))
    if SdmaDdrTotalCycle == 0:
        SdmaDdrAvgBandwidthList.append(0)
    else:
        SdmaDdrAvgBandwidthList.append(
            str((Decimal((SdmaDdrTotalDatasizeList[-1] / SdmaDdrTotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    if SdmaL2TotalCycle == 0:
        SdmaL2AvgBandwidthList.append(0)
    else:
        SdmaL2AvgBandwidthList.append(
            str((Decimal((SdmaL2TotalDatasizeList[-1] / SdmaL2TotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    if sdmaTotalXactCnt > 0:
        sdmaDdrAvgBurstLengthList.append(
            str((Decimal(sdmaTotalBurstLength / sdmaTotalXactCnt)).quantize(Decimal("0.00"))))
    else:
        sdmaDdrAvgBurstLengthList.append(0)
    #cdma
    SimCdmaCycleList.append(max(SimCdmaCycleList))
    CdmaDdrTotalDatasizeList.append(sum(CdmaDdrTotalDatasizeList))
    CdmaL2TotalDatasizeList.append(sum(CdmaL2TotalDatasizeList))
    if CdmaDdrTotalCycle == 0:
        CdmaDdrAvgBandwidthList.append(0)
    else:
        CdmaDdrAvgBandwidthList.append(
            str((Decimal((CdmaDdrTotalDatasizeList[-1] / CdmaDdrTotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    if CdmaL2TotalCycle == 0:
        CdmaL2AvgBandwidthList.append(0)
    else:
        CdmaL2AvgBandwidthList.append(
            str((Decimal((CdmaL2TotalDatasizeList[-1] / CdmaL2TotalCycle * frequency / 1000))).quantize(Decimal("0.00"))))
    if cdmaTotalXactCnt > 0:
        cdmaDdrAvgBurstLengthList.append(
            str((Decimal(cdmaTotalBurstLength / cdmaTotalXactCnt)).quantize(Decimal("0.00"))))
    else:
        cdmaDdrAvgBurstLengthList.append(0)
    
    CoreIdList = [str(i) for i in range(0, actualCoreNum)]
    CoreIdList.append('Overall')
    for rowIndex in [len(SimTotalCycleList) - 1]:
        SimTotalCycleList[rowIndex] = str(
            (Decimal(SimTotalCycleList[rowIndex] / frequency)).quantize(Decimal("0.00"))) + 'us'
        SimTiuCycleList[rowIndex] = str(
            (Decimal(SimTiuCycleList[rowIndex] / frequency)).quantize(Decimal("0.00"))) + 'us'
        SimGdmaCycleList[rowIndex] = str(
            (Decimal(SimGdmaCycleList[rowIndex] / frequency)).quantize(Decimal("0.00"))) + 'us'
        SimSdmaCycleList[rowIndex] = str(
            (Decimal(SimSdmaCycleList[rowIndex] / frequency)).quantize(Decimal("0.00"))) + 'us'
        SimCdmaCycleList[rowIndex] = str(
                (Decimal(SimCdmaCycleList[rowIndex] / frequency)).quantize(Decimal("0.00"))) + 'us'
    summaryInfoCols = ['CoreId', 'TiuWorkingRatio','Parallelism(%)','simTotalCycle','simTiuCycle', 'uArchURate',
                       'simGdmaCycle', 'GdmaDdrAvgBandwidth(GB/s)','GdmaL2AvgBandwidth(GB/s)', 'GdmaAvgDdrBurstLength']
    summaryData = [CoreIdList, TiuWorkingRatioList, ParallelismList, SimTotalCycleList, SimTiuCycleList,  uArchRateList,
                             SimGdmaCycleList,GdmaDdrAvgBandwidthList, GdmaL2AvgBandwidthList,gdmaDdrAvgBurstLengthList]
    if not sdmaDf.empty:
        summaryInfoCols.extend(['simSdmaCycle', 'SdmaDdrAvgBandwidth(GB/s)', 'SdmaAvgDdrBurstLength'])
        summaryData.extend([SimSdmaCycleList, SdmaDdrAvgBandwidthList, sdmaDdrAvgBurstLengthList])
    if not cdmaDf.empty:
        summaryInfoCols.extend(['simCdmaCycle', 'CdmaDdrAvgBandwidth(GB/s)','CdmaL2AvgBandwidth(GB/s)', 'CdmaAvgDdrBurstLength'])
        summaryData.extend([SimCdmaCycleList, CdmaDdrAvgBandwidthList, CdmaL2AvgBandwidthList,cdmaDdrAvgBurstLengthList])
    summaryData = transpose(summaryData).tolist()
    summaryDf = pd.DataFrame(summaryData, columns=summaryInfoCols, index=None)
    return alldata, summaryDf

def read_configs(file_path):
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def generate_lmem_partition(lmem_size, lane_num):
    lmem_partition = []
    partition_size = lmem_size // lane_num // 16
    start_value = 0

    for i in range(16):
        sublist = [start_value, partition_size, f'BANK[{i}]']
        lmem_partition.append(sublist)
        start_value += partition_size
    return lmem_partition

def multiply_non_zero_numbers(input_str):
    nums = input_str.split(')')[0][1:].split(',')
    result = reduce(lambda x, y: x * y, [int(n) for n in nums if int(n) != 0])
    return result

def deduplicate_ordered_list(lst):
    seen = set()
    deduped = []
    for item in lst:
        t_item = tuple(item)
        if t_item not in seen:
            seen.add(t_item)
            deduped.append(item)
    return deduped

def process_data(data, idx, data_type, ddrBw, read_directions, write_directions, lane_num, cycle_data_dict, lmem_op_dict):
    if data.empty:
        return
    if 'Bandwidth' in data:
        data['Bandwidth'] = data['Bandwidth'].apply(lambda x: str(x) if isinstance(x, Decimal) else x)
    bd = pd.to_numeric(data['Bandwidth']).apply(lambda x: round(x / ddrBw, 2)) if 'Bandwidth' in data else None

    for i in range(len(data)):
        uarch_rate = pd.to_numeric(data['uArch Rate'][i][:-1]) if 'uArch Rate' in data.columns else None
        tmp = [
            data_type,
            int(data['Start Cycle'][i]),
            int(data['End Cycle'][i]),
            int(data['End Cycle'][i]) - int(data['Start Cycle'][i]),
            data['Function Type'][i] if 'Function Type' in data else '',
            round(uarch_rate/100, 2) if uarch_rate is not None else bd[i],
            int(data['Cmd Id'][i]),
            data['Function Name'][i],
            data['Bandwidth'][i] if 'Bandwidth' in data else data['uArch Rate'][i],
            data['Data Type'][i],
            f"Direction:{data['Direction'][i]}" if 'Direction' in data else f"Bank Conflict Ratio:{data['Bank Conflict Ratio'][i]}",
            data['Msg Id'][i],
            data['Sd\Wt Count'][i],
        ]
        cycle_data_dict[f'time_data{idx}'].append(tmp)

        if 'Direction' in data:
            direction = data['Direction'][i]
            op_type = 0 if direction in read_directions else 1
            if direction in read_directions:
                size_key = 'dst_shape'
                addr_key = 'dst_start_addr'
                size = multiply_non_zero_numbers(data[size_key][i]) / lane_num
                # eval('*'.join([num for num in data[size_key][i].strip('()').split(',') if num != '0'])) / lane_num
                lmem_op_dict[f'lmem_op_record{idx}'].append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), op_type, data[addr_key][i], size, 'Direction: ' + direction])
            elif direction in write_directions:
                size_key = 'src_shape'
                addr_key = 'src_start_addr'
                size = multiply_non_zero_numbers(data[size_key][i]) / lane_num
                lmem_op_dict[f'lmem_op_record{idx}'].append([int(data['Start Cycle'][i]), int(data['End Cycle'][i]), op_type, data[addr_key][i], size, 'Direction: ' + direction])
    cycle_data_dict[f'time_data{idx}'] = deduplicate_ordered_list(cycle_data_dict[f'time_data{idx}'])
    lmem_op_dict[f'lmem_op_record{idx}'] = deduplicate_ordered_list(lmem_op_dict[f'lmem_op_record{idx}'])


def read_chipArchArgs(files_to_check):
    chipArchArgs = {}
    # 检查文件大小，找到不为空的文件
    linecount = 0
    for file_name in files_to_check:
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    linecount += 1
                    if "\t" in line:
                        fields = line.split(': ')
                        attr = fields[0][1:]
                        val = fields[1][:-1]
                        chipArchArgs[attr] = val
                    if '__CDMA_REG_INFO__' in line or '__TDMA_REG_INFO__' in line or '__TIU_REG_INFO__' in line:
                        break
            if chipArchArgs:
                break
    # print('chipArchArgs:',chipArchArgs)
    # print("linecount:",linecount)
    return chipArchArgs, linecount

def calculate_ratios(cycle_data_dict):
    ddr_ratios_dict = defaultdict(float)
    l2m_ratios_dict = defaultdict(float)
    prev_ddr_ratio, prev_l2m_ratio = None, None
    ddr_ratios, l2m_ratios = [], []

    for data in cycle_data_dict.values():
        # 使用生成器表达式进行过滤
        filtered_data = (
            record for record in data
            if record[0] in [1, 2] and "SYS" not in record[4]
        )

        for record in filtered_data:
            category, begin_time, end_time, _, _, _, _, _, uarch_bw, _, info, _, _ = record

            for time in range(begin_time + 1, end_time + 1):
                bw = float(uarch_bw)
                if 'DDR' in info and bw:
                    ddr_ratios_dict[time] += bw
                if 'L2M' in info and bw:
                    l2m_ratios_dict[time] += bw

    for time, bw in sorted(ddr_ratios_dict.items()):
        new_ddr_ratio = bw / 546
        if new_ddr_ratio != prev_ddr_ratio:
            ddr_ratios.append([time, new_ddr_ratio])
            prev_ddr_ratio = new_ddr_ratio

    for time, bw in sorted(l2m_ratios_dict.items()):
        new_l2m_ratio = bw / 1024
        if new_l2m_ratio != prev_l2m_ratio:
            l2m_ratios.append([time, new_l2m_ratio])
            prev_l2m_ratio = new_l2m_ratio

    return ddr_ratios, l2m_ratios

def generate_html(dirpath, name):
    regInfo_files = [f"{dirpath}/tiuRegInfo_0.txt", f"{dirpath}/tdmaRegInfo_0.txt", f"{dirpath}/cdmaRegInfo_0.txt"]
    chipArchArgs, linecount = read_chipArchArgs(regInfo_files)
    global CHIP_ARCH
    CHIP_ARCH = chipArchArgs['Chip Arch']
    print("CHIP_ARCH:",CHIP_ARCH)
    core_num = int(chipArchArgs['Core Num'])
    ddrBw = pd.to_numeric(chipArchArgs['DDR Max BW(GB/s)'])
    L2Bw = pd.to_numeric(chipArchArgs['L2 Max BW(GB/s)'])
    #ddrBw = configs['LANE_NUM'] * configs['FREQUENCY'] * 64 / 8000 / 1000 #A2
    if core_num <= 2:
        categories = ["TPU_BD", "TPU_GDMA"]
    else:
        categories = ["TPU_BD", "TPU_GDMA", "TPU_SDMA", "TPU_CDMA"]

    lmem_size = int(chipArchArgs['Tpu Lmem Size']) #configs['TPU_LOCAL_MEMORY_SIZE']
    lane_num = int(chipArchArgs['NPU Num']) #configs['LANE_NUM']
    lmem_partition = generate_lmem_partition(lmem_size,lane_num)

    simulatorFile = os.path.join(dirpath, 'simulatorTotalCycle.txt')
    tiuRegFile =  os.path.join(dirpath,'tiuRegInfo')
    gdmaRegFile =  os.path.join(dirpath, 'tdmaRegInfo')
    cdmaRegFile =  os.path.join(dirpath, 'cdmaRegInfo')
    alldata, summarydf = RegInfo(tiuRegFile, gdmaRegFile, cdmaRegFile, simulatorFile, core_num, chipArchArgs, linecount)
    summary_data =[[str(x) if isinstance(x,Decimal) else x for x in lst] for lst in summarydf.values.tolist()]

    n = len(alldata)
    cycle_data_dict = {f"time_data{i}": [] for i in range(0, n)}
    lmem_op_dict = {f"lmem_op_record{i}": [] for i in range(0, n)}

    for idx in range(n):
        read_directions = ['DDR->LMEM'] + [f'DDR->LMEM{i}' for i in range(8)] + [f'L2M->LMEM{i}' for i in range(8)]
        write_directions = ['LMEM->DDR'] + [f'LMEM{i}->DDR' for i in range(8)] + [f'LMEM{i}->L2M' for i in range(8)]
        tiudata = alldata[idx][0]
        process_data(tiudata, idx, 0, ddrBw, read_directions, write_directions, lane_num, cycle_data_dict, lmem_op_dict)
        dmadata = alldata[idx][1]
        process_data(dmadata, idx, 1, ddrBw, read_directions, write_directions, lane_num, cycle_data_dict, lmem_op_dict)
        sdmadata = alldata[idx][2]
        process_data(sdmadata, idx, 2, ddrBw, read_directions, write_directions, lane_num, cycle_data_dict, lmem_op_dict)
        cdmadata = alldata[idx][3]
        process_data(cdmadata, idx, 3, ddrBw, read_directions, write_directions, lane_num, cycle_data_dict, lmem_op_dict)

    if CHIP_ARCH == 'sg2260':
        ddr_ratios, l2m_ratios = calculate_ratios(cycle_data_dict)
    else:
        ddr_ratios, l2m_ratios = [], []
    data_filepath = f"{out_path}/profile_data.js"
    # page_caption = f"PerfAI: {name}"
    # page_caption += name
    with open(data_filepath, "w") as js:
        js.write(f'let page_caption = "PerfAI: {name}"\n')
        js.write(f'let platform = "Platform: {CHIP_ARCH}"\n')
        js.write(f'let configs = {chipArchArgs}\n')
        js.write('let summary_caption= "Summary Table"\n')
        js.write(f'let summary_header =  {summarydf.columns.tolist()}\n')
        js.write(f'let summary_data = {summary_data}\n')
        js.write(f'let ddr_bandwidth = {ddrBw}\n')
        js.write(f'let l2_bandwidth = {L2Bw}\n')
        js.write(f'let ddr_ratios = {ddr_ratios}\n')
        js.write(f'let l2m_ratios = {l2m_ratios}\n')
        js.write(f'let categories = {categories}\n')
        time_header = ["category", "begin_time(cycle)", "end_time(cycle)", "Duration(cycles)", "func_type", "height", "cmd", "func_name", "uArchRate/BW", "Data Type", "Info","Msg_Id","Sd/Wt_Count"]
        filter_cols = [time_header.index(c) for c in ["category", "func_type"]]
        js.write(f'let filter_cols = {filter_cols}\n')
        js.write(f'let lmem_partition = {lmem_partition}\n')
        js.write(f'let time_header = {time_header}\n')
        for lmem_op in lmem_op_dict.keys():
            js_content = ""
            for i, sublist in enumerate(lmem_op_dict[lmem_op]):
                js_content += f"{sublist},\n"
            js.write(f'window.{lmem_op} = [{js_content}]\n')

        for keyname in cycle_data_dict.keys():
            js_content = ""
            for i, sublist in enumerate(cycle_data_dict[keyname]):
                js_content += f"{sublist},\n"
            js.write(f'window.{keyname} = [{js_content}]\n')

import time
if __name__ == '__main__':
    """
    The html and echarts.min.js should be first put under the same main dir
    python perfAI_html.py output_dir title_name
    """
    start = time.time()
    args = sys.argv
    out_path = os.path.join(args[1], 'PerfWeb')
    os.makedirs(out_path,exist_ok=True)
    file_path = args[0][:args[0].rfind('/')]
    htmlfiles = [f'{file_path}/echarts.min.js', f'{file_path}/jquery-3.5.1.min.js', f'{file_path}/result_test.html']
    for f in htmlfiles:
        shutil.copy2(f, out_path)
    print("Start generating data!")
    generate_html(args[1], out_path)
    end = time.time()
    passed = end-start
    print(f"Total spent time: {passed}seconds")
    print(f"The jsfile is generated successfully under {out_path}")