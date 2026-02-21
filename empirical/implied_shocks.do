clear all
qui cd "C:\Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)\Documents\GitHub computer 2\NoShort\empirical\"

local countries "US Finland Germany Norway"

foreach country of local countries {
	import excel "realized_shocks_`country'.xlsx", sheet("Sheet1") firstrow clear
	gen mon = mod(yyyymm, 100)
	gen Year = int(yyyymm / 100)
	drop if Year <= 1920
	gen Yearmon = ym(Year, mon)
	sort Yearmon 
	gen Z_`country' = sum(dZ)
	gen ZSI_`country' = sum(dZ_SI)
	drop dZ dZ_SI yyyymm
	save "realized_shocks_`country'.dta", replace
}

local countries "Finland Germany Norway"
use "realized_shocks_US.dta", clear
foreach country of local countries {
	merge 1:1 Yearmon using "realized_shocks_`country'.dta", nogenerate
}	
save "realized_shocks_all.dta", replace

reshape long Z_ ZSI_, i(Yearmon) j(country) str
replace ZSI = . if ZSI == 0

label variable Z_ "Shocks to fundamental"
label variable ZSI_ "Shocks to signal"

egen id = group(country)
xtset id Yearmon, monthly

twoway (tsline Z_, lcolor(dknavy) lwidth(medium)) (tsline ZSI_, lcolor(maroon) lwidth(medium) lpattern(vshortdash)) if Year >= 1920, ytitle("Implied Shocks") ylabel(, nogrid) ttitle("Year") tlabel(, labsize(small) nogrid) by(, legend(position(6))) legend(rows(1) size(small)) xsize(25) ysize(15) by(country, yrescale)

graph export "time-series-shocks.png", as(png) name("Graph") replace
