def get_phase_and_blending(iter, opt):
    transition_img = opt.lod_transition_img
    train_img = opt.lod_train_img
    phase_dur = transition_img + train_img
    phase_idx = iter // phase_dur
    phase_img = iter - phase_idx * phase_dur

    if (phase_img < train_img) and (phase_idx+1 <= opt.num_phases):
        alpha = 1
    elif (phase_idx+1 <= opt.num_phases):
        alpha = 1 - (phase_img - train_img) / transition_img
    else:
        alpha = 0

    phase = phase_idx + 1 if phase_idx + 1 <= opt.num_phases else opt.num_phases
    assert 0 <= alpha <= 1

    target_res = 's' + str(2 ** (opt.num_phases - phase))

    return phase, alpha, target_res
