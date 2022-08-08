from blockchain import blockexplorer

def get_transaction(inputs, outputs):
    inputs_addrs = []
    outputs_addrs = []
    inputs_vals = []
    outputs_vals = []
    for i in inputs:
        if 'address' in dir(i) and 'value' in dir(i):
            inputs_addrs.append(i.address)
            inputs_vals.append(i.value)
    for j in outputs:
        if 'address' in dir(j) and 'value' in dir(j):
            outputs_addrs.append(j.address)
            outputs_vals.append(j.value)
    return inputs_addrs, inputs_vals, outputs_addrs, outputs_vals

if __name__ == "__main__":
    block_hash = '0000000000000000000067c2b9304f6bdded7d6c1f58cc0ec089617d67445a56'
   
    Flag = False
    print(block_hash)
    while(not Flag):
        try:
            block = blockexplorer.get_block(block_hash)
            Flag = True
        except Exception as err:
            print(str(err))
        
    transactions=block.transactions
    trans_hash=[]
    trans_inputs=[]
    trans_outputs=[]
    for transaction in transactions:
        trans_hash.append(transaction.hash)
        trans_inputs.append(transaction.inputs)
        trans_outputs.append(transaction.outputs)
    for i in range(len(trans_hash)):
        inputs_addrs, inputs_vals, outputs_addrs, outputs_vals = get_transaction(trans_inputs[i], trans_outputs[i])
        # inputs_sum = 0.0
        # outputs_sum = 0.0
        sum1=0.0
        sum2=0.0
        for val in inputs_vals:
            #inputs_sum += float(val)
            sum1 += float(val)
        for val in outputs_vals:
            #outputs_sum += float(val)
            sum2 += float(val)
        for i, addr_in in enumerate(inputs_addrs):
            for j, addr_out in enumerate(outputs_addrs):
                val = float(inputs_vals[i]) / sum1 * outputs_vals[j]
                # if val == 0.0:
                #     continue
                      
                val = str(val)
                addr_in = str(addr_in)
                addr_out = str(addr_out)
                with open('data.csv', 'a+') as f:
                    f.write(addr_in + ',' + addr_out + ',' + val + '\n')            
    print('Already got data')




