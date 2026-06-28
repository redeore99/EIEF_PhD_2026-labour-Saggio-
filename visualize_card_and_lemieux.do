use id year log_dailywages age female using "/Users/raffaelesaggio/Documents/GitHub/LeaveOutTwoWay/data/py_final_1985_2001_veneto_only.dta",clear



*age restrictions and age groups
drop if age<=25 | age>60
gen age_group = 0 if age>25 & age<=30
replace age_group = 1 if age>30 & age<=35
replace age_group = 2 if age>35 & age<=40
replace age_group = 3 if age>40 & age<=45
replace age_group = 4 if age>45 & age<=50
replace age_group = 5 if age>50 & age<=55
replace age_group = 6 if age>55 & age<=60

gen Q=1

collapse log_dailywages (sum) Q, by(age_group year female)
replace Q = log(Q)

bys female year: egen norm   = mean(log_dailywages) if age_group == 0
bys female year: egen norm_y = mean(norm)
drop norm
bys female year: egen norm   = mean(Q) if age_group == 0
bys female year: egen norm_x = mean(norm)

gen y = log_dailywages-norm_y
gen x = Q-norm_x

reghdfe y x if female == 0, abs(year age_group)
reghdfe y x if female == 1, abs(year age_group)








