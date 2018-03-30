from datetime import datetime


def get_filename(score=0.0, algorithm_name=None):

    currentTime = datetime.now();
    currentMinute = currentTime.minute
    currentHour = currentTime.hour

    currentDay = currentTime.day
    currentMonth = currentTime.month
    currentYear = currentTime.year
    filename = str(currentYear)+'_'+\
              str(currentMonth)+'_'+\
              str(currentDay)+'_'+\
              str(currentHour)+'_'\
              +str(currentMinute)+'__'+\
           '{:2.2f}'.format(score*100)

    if algorithm_name == None:
       return filename
    else:
        return filename+'__'+algorithm_name



