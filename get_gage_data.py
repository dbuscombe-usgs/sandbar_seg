def generate_url(name_of_station):

    url1 = 'http://waterservices.usgs.gov/nwis/iv/?'
    url2 = 'format=rdb'
    url3 = 'sites=' + name_of_station
    url4 = 'startDT=2015-01-01'
    url5 = 'endDT=2016-01-01'
    url6 = 'parameterCd=00060,00065'

    url = url1 + url2 + '&' + url3 + '&' + url4 + '&' + url5 + '&' + url6
    
    return url


lees_ferry = '09380000'
grand_canyon = '09402500'
diamond_creek = '09404200'

generate_url(lees_ferry)
