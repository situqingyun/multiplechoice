import torch
import torch.functional as F


def concat_all_encoders_hidden_states(all_encoder_layers, rnn, linears):
    all_encoder_layers=all_encoder_layers[1:]
    output_list = list()
    for i in range(len(all_encoder_layers)):
        output, (final_hidden_state, final_cell_state) = rnn(all_encoder_layers[i])
        output_list.append(linears[i](output))

    output_tensor = torch.cat(output_list, 2)
    output_tensor = F.softmax(output_tensor, 2)

    all_layers = torch.cat([torch.unsqueeze(i, 2) for i in all_encoder_layers], axis=2)  # 第三维度拼接
    focus = torch.matmul(torch.unsqueeze(output_tensor, axis=2), all_layers)

    sequence_output = torch.squeeze(focus, 2)
    return sequence_output

