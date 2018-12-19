def get_phase_and_blending(iter, opt):
    transition_img = opt.lod_transition_img
    train_img = opt.lod_train_img
    phase_dur = transition_img + train_img
    phase_idx = iter // phase_dur
    phase_img = iter - phase_idx * phase_dur

    phase = phase_idx + 1 if phase_idx + 1 <= opt.num_phases else opt.num_phases
    if phase_img < train_img:
        alpha = 1
    else:
        alpha = 1 - (phase_img - train_img) / transition_img

    assert 0 <= alpha <= 1

    return phase, alpha
