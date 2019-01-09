from util.training_schedule import get_phase_and_blending

class mock_options:

    def __init__(self):
        self.lod_transition_img = 10
        self.lod_train_img = 10
        self.num_phases = 2


opt = mock_options()
for i in range(110):
    phase, alpha = get_phase_and_blending(i, opt)
    target_res = 2**(opt.num_phases-phase)
    print('phase {}, alpha {}, target res: {}'.format(phase, alpha, target_res))
