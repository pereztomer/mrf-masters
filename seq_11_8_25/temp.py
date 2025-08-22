number_of_y_blips_to_center = round(abs(gp_pre.area) / gp_blip.area)

epi_step_time = (pp.calc_duration(gx, adc)
                 + pp.calc_duration(gp_blip)
                 + pp.calc_duration(gx_, adc)
                 + pp.calc_duration(gp_blip))

if number_of_y_blips_to_center % 2 == 0:
    term_2 = number_of_y_blips_to_center / 2 * epi_step_time
    term_2 += pp.calc_duration(gp_blip) + pp.calc_duration(gx_, adc) / 2

if number_of_y_blips_to_center % 2 == 1:
    term_2 = (number_of_y_blips_to_center - 1) / 2 * epi_step_time
    term_2 += 2 * pp.calc_duration(gp_blip) + pp.calc_duration(gx_, adc) + pp.calc_duration(gx_pre, adc) / 2

term_2 = term_2 + pp.calc_duration(rf_pulses[0], gz_list[0]) + gx_pre, gp_pre, gz_reph_list[0]