#include "main.h"
#include <arm_neon.h>
#include "math/fast_prng/Xoroshiro128plus_vectorized.h"
#include "math/functions/vectorized_exp_log.h"
#include "math/functions/vectorized_trig.h"
#include "math/functions/vectorized_trig_taylor.h"
#include "math/ziggurat/normal.h"
#include "math/entropy.h"

using namespace std;

void initialize() {
	pros::c::serctl(SERCTL_DISABLE_COBS,NULL);

	// pros::lcd::register_btn1_cb(on_center_button);
}

void disabled() {}

void competition_initialize() {}

void autonomous() {}

void opcontrol() {
    Vuniform_int32_t rand_int_gen(1,100, 1230);
    cout << "testing random ints in the range [1, 100]\n";
    for(int i = 0; i < 10;i++){
        int32x4_t current = rand_int_gen();
        cout << "current: " << endl;
        cout << vgetq_lane_s32(current,0) << endl;
        cout << vgetq_lane_s32(current,1) << endl;
        cout << vgetq_lane_s32(current,2) << endl;
        cout << vgetq_lane_s32(current,3) << endl;
    }
    Vuniform_float32_t rand_float_gen(2.5,2.75, 1230);
    cout << "testing random ints in the range [2.5,2.75)\n";
    for(int i = 0; i < 10;i++){
        float32x4_t current = rand_float_gen();
        cout << "current: " << endl;
        cout << vgetq_lane_f32(current,0) << endl;
        cout << vgetq_lane_f32(current,1) << endl;
        cout << vgetq_lane_f32(current,2) << endl;
        cout << vgetq_lane_f32(current,3) << endl;
    }

    VXoroshiro128plus rand_gen(1230);

    for(int i = 0; i < 10;i++){
        uint32x4_t current = rand_gen.next();
        cout << "current" << endl;
        cout << vgetq_lane_u32(current,0) << endl;
        cout << vgetq_lane_u32(current,1) << endl;
        cout << vgetq_lane_u32(current,2) << endl;
        cout << vgetq_lane_u32(current,3) << endl;
    }

    cout << endl << "now testing trig" << endl;
    const v4sf trig_test = {0,M_PI/6,M_PI/4,M_PI/3};
    v4sf sin_res, cos_res;

    sincos_ps(trig_test, &sin_res, &cos_res);
    cout << "cosines\n";
    cout << vgetq_lane_f32(cos_res,0) << endl;
    cout << vgetq_lane_f32(cos_res,1) << endl;
    cout << vgetq_lane_f32(cos_res,2) << endl;
    cout << vgetq_lane_f32(cos_res,3) << endl;
    cout << "sines\n";
    cout << vgetq_lane_f32(sin_res,0) << endl;
    cout << vgetq_lane_f32(sin_res,1) << endl;
    cout << vgetq_lane_f32(sin_res,2) << endl;
    cout << vgetq_lane_f32(sin_res,3) << endl;

    cout << endl << "now testing taylor trig" << endl;
    const v4sf t_trig_test = {M_PI/6, M_PI/4, M_PI/2, 3*M_PI/4};
    const v4sf t_c = {M_PI/2,M_PI/2,M_PI/2,M_PI/2};

    const v4sf csin = {1,1,1,1};
    const v4sf ccos = {0,0,0,0};
    v4sf t_sin_res, t_cos_res;

    Vsincos_taylor(t_trig_test, t_c,csin,ccos, &t_sin_res, &t_cos_res);
    cout << "cosine errors with c = pi/2 :\n";
    cout << "pi/6 " <<  vgetq_lane_f32(t_cos_res,0) - cos(M_PI/6) << endl;
    cout << "pi/4 " << vgetq_lane_f32(t_cos_res,1) - cos(M_PI/4) << endl;
    cout << "pi/2 " << vgetq_lane_f32(t_cos_res,2) - cos(M_PI/2) << endl;
    cout << "3pi/4 "<< vgetq_lane_f32(t_cos_res,3) - cos(3*M_PI/4) << endl;
    cout << "sine errors with c = pi/2 :\n";
    cout << "pi/6 " << vgetq_lane_f32(t_sin_res,0) - sin(M_PI/6) << endl;
    cout << "pi/4 " << vgetq_lane_f32(t_sin_res,1) - sin(M_PI/4) << endl;
    cout << "pi/2 " << vgetq_lane_f32(t_sin_res,2) - sin(M_PI/2) << endl;
    cout << "3pi/4 "<< vgetq_lane_f32(t_sin_res,3) - sin(3*M_PI/4) << endl;

    cout << "testing normal nums\n";
    RobotEntropy<uint32_t> rng;
    math::ziggurat::NormalPRNG normal_gen(rng());
    for(int i = 0;i < 10;i++){
        float num = normal_gen.normal(6,1);
        cout << num << '\n';
    }
}
