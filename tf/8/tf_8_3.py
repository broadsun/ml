#coding=utf8

lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MMultiRNNCell([lstm] * number_of_layers)

state = stacked_lstm.zero_state(batch_size, tf.float32)

for i in range(len(num_steps)):
    if i > 0:
        tf.get_variable_scope().reuse_variable()
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fully_connected(stacked_lstm_output)
    loss += calc_loss(final_output, expected_output)


lstm = rnn_cell.BasicLSTMCell(lstm_size)
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob = 0.5)

stacked_lstm = rnn_cell.MMultiRNNCell([dropout_lstm] * number_of_layers)

