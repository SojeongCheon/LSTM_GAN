# import tensorboard as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

summary_writer = tf.summary.FileWriter("/home/sojeong/LSTM_Synthesis2/log/2020_12_04_15_02_0th_fold_nodule100")

cnt = 0

for event in tf.train.summary_iterator("/home/sojeong/LSTM_Synthesis2/log/10 fold cross validation/SequentialSynthetic_2020_12_04_15_02/events.out.tfevents.1607090566.kisteu-DGX-Station.4473.0"):   ### 38850
    break_tag = 0    
    summary = tf.Summary()

    for idx, value in enumerate(event.summary.value):
        if value.tag == 'test_loss/test_err_background':
            cnt += 1
        
        # if cnt < 301 :
        #     break_tag = 1
        #     break
        summary.value.add(tag='{}'.format(value.tag),simple_value=value.simple_value)
    
    # if break_tag :
    #     continue


    summary_writer.add_summary(summary, event.step)
    summary_writer.flush()

    if cnt == 300:
        break
print(cnt)

