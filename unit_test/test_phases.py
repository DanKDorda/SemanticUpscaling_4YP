from util.training_schedule import get_phase_and_blending

class mock_options:

    def __init__(self):
        self.lod_transition_img = 5
        self.lod_train_img = 5
        self.num_phases = 6


opt = mock_options()
for i in range(130):
    phase, alpha, target_res = get_phase_and_blending(i, opt)
    print('phase {}, alpha {}, target res: {}'.format(phase, alpha, target_res))
