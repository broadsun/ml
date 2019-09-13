#coding=utf8

lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)
state = lstm.zero_state(batch_size, tf.float32)

loss = 0.0
for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variable()
    lstm_output, state = lstm(current_input, state)
    final_output = fully_connected(lstm_output)
    loss += calc_loss(final_output, expected_output)

