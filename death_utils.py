

def get_dmf(num=1):
    dmf = []
    count = 0
    with open('ssdm' + str(num), 'r') as f:
        for item in f:
            dmf.append(item)
            count += 1
            if count % 1000 == 0:
                print(str(int((count * 100) / 28700000)) + '%\r', end='')
    print('got ' + str(len(dmf)) + ' entries')
    return dmf

def make_entry(s):
    entry = {}
    if ',' in s:
        s = s.replace(',',' ')
    entry['ssn'] = s[1:10]
    entry['firstname'] = s[34:49]
    entry['lastname'] = s[10:30]
    entry['middlename'] = s[49:64]
    entry['suffix'] = s[30:34]
    entry['dob'] = s[73:75] + '-' + s[75:77] + '-' + s[77:81] 
    entry['dod'] = s[65:67] + '-' + s[67:69] + '-' + s[69:73]
    entry['age'] = int(s[69:73]) - int(s[77:81])
    if int(s[65:67]) < int(s[73:75]) and int(s[65:67]) != 0 and int(s[73:75]) != 0:
        entry['age'] -= 1
    elif int(s[65:67]) == int(s[73:75]) and int(s[67:69]) < int(s[75:77]) and int(s[65:67]) != 0 and int(s[75:77]) != 0:
        entry['age'] -= 1
    if int(s[69:73]) == 0 or int(s[77:81]) == 0:
        entry['age'] = float('nan')
    return entry

def get_names_table(dmf, names):
    for s in dmf:
        entry = make_entry(s)
        firstname = entry['firstname'].strip().lower()
        if firstname not in names:
            names[firstname] = []
        names[firstname].append(entry)
    return names

def get_avg_ages(names):
    THRESHOLD = 0
    avg_age = Series()
    for key in names.keys():
        count = 0
        total = 0
        for entry in names[key]:
            total += entry['age']
            count += 1
        if count > THRESHOLD:
            avg_age[key] = total / count
    return avg_age

#pass this function a pandas dataframe from reading the .csv created by write_state().
#this function aggregates into a list where each entry is a name followed by all ages of
#the deceased person who bore that name.
def state_dataframe_to_ages_master(state):
    count = 0
    names = {}
    fmaster_r = open('AGES_MASTER', 'r')
    fmaster_w = open('AGES_MASTER_temp', 'w')
    for item in fmaster_r:
        entry = item.split(',') #first,age,age,...,age
        names[entry[0]] = ','.join(entry[1:]).replace('\n', '')
        count += 1
        if count % 11 == 0:
            print('loaded %d entries from master file\r' % count, end='')
    count = 0
    print('\naggregating', end='')
    for person in state.iterrows():
        if person[1]['first'] not in names:
            names[person[1]['first']] = str(person[1]['age'])
        else:
            names[person[1]['first']] += ',' + str(person[1]['age'])
        count += 1
        if count % 1000 == 0:
            print('\raggregating names %.2f percent' % float(count * 100 / len(state)), end='')
    count = 0
    for name in names.keys():
        fmaster_w.write(str(name) + ',' + names[name] + '\n')
        count += 1
        if count % 100 == 0:
            print('writing names %.2f percent\r' % float(count * 100 / len(names)), end='')
    fmaster_w.close()
    fmaster_r.close()
    fmaster_w = open('AGES_MASTER', 'w')
    fmaster_r = open('AGES_MASTER_temp', 'r')
    fmaster_w.write(fmaster_r.read())
    fmaster_w.close()
    fmaster_r.close()
    print('wrote %d name entries to AGES_MASTER                     ' % len(names))

#pass this function a pandas dataframe from reading the .csv created by write_state().
#this function aggregates into a list where each entry is a name followed by a section
#for each person separated by commas in the format BBBB:DDDD:AAA, corresponding to 
#birth year, death year, and age. Age will be one to three characters long.
def state_dataframe_to_birth_death_master(state):
    count = 0
    nan_count = 0
    names = {}
    import math
    try:
        fmaster_r = open('BIRTH_DEATH_MASTER', 'r') 
        for item in fmaster_r:
            entry = item.split(',') #first,section, section, ... section
            names[entry[0]] = ','.join(entry[1:]).replace('\n', '')
            count += 1
            if count % 11 == 0:
                print('loaded %d entries from master file\r' % count, end='')
    except:
        pass       
    fmaster_w = open('BIRTH_DEATH_temp', 'w')
    count = 0
    print('\naggregating', end='')
    for person in state.iterrows():
        if not math.isnan(person[1]['age']):
            if person[1]['first'] not in names:
                names[person[1]['first']] = str(person[1]['dob'][6:]) + ':' + str(person[1]['dod'][6:]) + ':' + str(int(person[1]['age']))
            else:
                names[person[1]['first']] += ',' + str(person[1]['dob'][6:]) + ':' + str(person[1]['dod'][6:]) + ':' + str(int(person[1]['age']))
            count += 1
        else:
            nan_count += 1
        if count % 1000 == 0:
            print('\raggregating names %.2f percent' % float(count * 100 / len(state)), end='')
    count = 0
    for name in names.keys():
        fmaster_w.write(str(name) + ',' + names[name] + '\n')
        count += 1
        if count % 100 == 0:
            print('writing names %.2f percent\r' % float(count * 100 / len(names)), end='')
    fmaster_w.close()
    try: fmaster_r.close() 
    except: pass
    print('copying from temp to master file...                         ')
    fmaster_w = open('BIRTH_DEATH_MASTER', 'w')
    fmaster_r = open('BIRTH_DEATH_temp', 'r')
    fmaster_w.write(fmaster_r.read())
    fmaster_w.close()
    fmaster_r.close()
    print('wrote %d name entries to BIRTH_DEATH_MASTER                     ' % len(names))
    print('dropped %d NaN age entries' % nan_count)


def dmf_to_csv(fname):
    count = 0
    fin = open(fname, 'r')
    fout = open(fname + '.csv', 'w')
    fout.write('first name,middle name,last name,suffix,age,dob,dod,ssn\n')
    for person in fin:
        entry = make_entry(person)
        fout.write(entry['firstname'] + ',' + entry['middlename'] + ',' + entry['lastname'] + ',' + entry['suffix'] + ',' + str(entry['age']) + ',' + entry['dob'] + ',' + entry['dod'] + ',' + entry['ssn'] + '\n')
        count += 1
        if count % 10000 == 0:
            print(str(count))
    fin.close()
    fout.close()

#writes from an array containing contiguous entries from the DMF (one of the DMF disks) 
#into a file for that particular state, defined by  the first three SSN digits. Starting in 
#June 2011 the first three digits are no longer used for this purpose.
def write_state(state, dmf):
    count = 0
    name = state['location'][state.index[0]].strip()
    f = open(name, 'w')
    start = state['start_idx'][state.index[0]]
    end = state['end_idx'][state.index[0]]
    print('\t\t\t' + name.upper(), '\n## printing every 10,000th entry ##')
    text = 'first,MI,last,suffix,age,dob,dod,ssn'
    f.write(text + '\n')
    print(text)
    total_len = len(dmf[start:end])
    for item in dmf[start:end]:
        entry = make_entry(item)
        text = entry['firstname'].strip() + ',' + entry['middlename'].strip() + ',' + entry['lastname'].strip() + ',' + entry['suffix'].strip() + ',' + str(entry['age']) + ',' + entry['dob'] + ',' + entry['dod'] + ',' + str(entry['ssn'])
        f.write(text + '\n')
        count += 1
        if count % 10000 == 0:
            print(text)
        if count % 1000 == 0:
            print(str(int(count * 100 / total_len)) + '%\r', end='')
    f.close()
    if total_len < 1000000:
        #print(str(int(total_len / 1000)) + 'k records written to "' + name + '.csv"')
        print('%dk records written to "%s.csv"' % (int(total_len / 1000), name))
    else:
        #print(str(int(total_len / 1000000)) + 'm records written to "' + name + '.csv"')
        print('%.2f records written to "%s.csv"' % (total_len / 1000000, name))

#pass this the file created by state_dataframe_to_ages_master()
def get_avg_age_at_death(fmaster):
    names = {}
    for item in fmaster:       
        total = 0
        count = 0
        entry = item.split(',')
        for age in entry[1:]:
            if not math.isnan(float(age.replace('\n',''))):
                total += float(age.replace('\n',''))
                count += 1
        if count > 0:
            names[entry[0]] = (total / count, count)
    return names


#process birth records into a single nested dictionary matching get_death_names(). the entry
#in dic[name][year] is the number born for that name-year combination.
def get_birth_names():
    year = 1880
    birth_names = {}
    while year < 2011:
        f = open('yob' + str(year) + '.txt', 'r')
        for line in f:
            entry = line.split(',')
            name = entry[0].lower()
            if name in birth_names:
                if year in birth_names[name]:
                    birth_names[name][year] += int(entry[2].strip())
                else:
                    birth_names[name][year] = int(entry[2].strip())
            else:
                birth_names[name] = {}
        year += 1
    return birth_names

#pass this the file output of state_dataframe_to_birth_death_master(); outputs a nested 
#dictionary to match get_birth_names(). the entry in dic[name][year] is a list of the
#ages at death for that name-year combination.
def get_death_names(f):
    names={}
    count=0
    for line in f:
        entry = line.split(',')
        name = entry[0].lower()
        names[name] = {}
        for s in entry[1:]:
            record = s.split(':')
            birth_year = int(record[0])
            age_at_death = record[2].strip()
            if int(age_at_death) > 0: #just throw out the several negative ages; hopefully it's the data
                if birth_year in names[name]:
                    names[name][birth_year] += ',' + age_at_death
                else:
                    names[name][birth_year] = age_at_death
        count += 1
        print('\r%d records' % count,end='')
    return names

#pass the dictionary from get_name_death_dictionary()
def plot_average_age(name, dic):
    x = []
    y = []
    for year in sorted(names[name].keys()):
        count = 0
        total = 0        
        x.append(year)
        for age in names[name][year].split(','):
            count += 1
            total += int(age)
        y.append(total/count)
    try:
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.show()
    except:
        print('Matplotlib not found, returning arrays')
        return (x,y)

def plot_births_from_birth_and_death_records(name, births, deaths_len, drop_before=1920):
    from pandas import Series
    #drop all years prior to given year from death records to match birth records
    birth_list = Series(births[name])
    death_list = Series(deaths_len[name])
    death_list.ix[drop_before:].plot(style='r-', label='SSA Death Master File')
    birth_list.ix[drop_before:].plot(style='b-', label='SSA Birth records')
    plt.ylabel('Birth count')
    plt.xlabel('Year')
    plt.title('People born with the name ' + name.capitalize() + ', by year')
    plt.legend(bbox_to_anchor=(0.,.93,1.,.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def compute_curve(name, year, bnames, dnames):
    try:
        dnames != None
        bnames != None
    except:
        print('birth and death records not found')
        Exception()
    try:
        dnames[name][year]
        bnames[name][year]
    except:
        return ''
    curve = [0] * 120
    for age in dnames[name][year].split(','):
        aad = int(age.strip())
        if aad > 0:
            curve[aad] += 1
    s = str(year) + ',' + str(bnames[name][year])
    oldest = 0
    for i in range(120):
        if curve[i] > 0 and i > oldest:
            oldest = i
    s += ',' + ','.join(map(str, curve[:oldest+1]))
    return s

def compute_all_curves(name):
    s = ''
    try:
        bnames[name]
    except:
        return ''
    for year in range(1920, 2011):
        s += ';' + compute_curve(name, year, bnames, dnames)
    if s.strip(';') == '':
        return ''
    return name + ';' + s + '\n'

#format: yyyy, number born this year, died at age 0, died at age 1, ..., died at oldest age
def plot_mortality_curve(curve):
    curve = curve.split(',')
    if len(curve) < 2:
        return
    from pandas import Series
    series = Series(curve[2:]).astype('int').cumsum() / int(curve[1])
    series.plot(label=curve[0])

def get_mortality_curves_from_file(fname):
    with open(fname,'r') as f:
        for line in f:
            entry = line.split(';')
            curves[entry[0]] = entry[1:]
    return curves

def compute_and_plot_all_mortality_curves(name):
    entry = compute_all_curves(name).split(';')
    for curve in entry[1:]:
        plot_mortality_curve(curve)
    plt.title('Mortality curves for ' + name.capitalize())
    plt.xlabel('Age')
    plt.ylabel('Fraction deceased')
    plt.show()

def plot_all_mortality_curves(entry):
    entry = entry.split(';')
    for curve in entry[1:]:
        plot_mortality_curve(curve)
    plt.title('Mortality curves for ' + entry[0].capitalize())
    plt.xlabel('Age')
    plt.ylabel('Fraction deceased')
    plt.show()

def plot_all_mortality_curves_from_file(f, limit=2000, threshold=1000, begin=1960):
    count = 0
    for line in f:
        entry = line.split(';')
        for curve in entry[begin-1919:]:
            if curve.strip() != '':
                second_comma = curve[5:].find(',') + 5
                if int(curve[5:second_comma]) > threshold:
                    plot_mortality_curve(curve)
        count += 1
        print('\r%d entries' % count, end='')
        if count > limit:
            break

def get_mortality_rate(curve, normalize=False, string=True, as_series=False):
    if string:
        curve = curve.split(',')
    if len(curve) < 3:
        return None
    from pandas import Series
    if normalize:
        series = Series(curve[2:]).astype('float') / int(curve[1]) * 1000
    else:
        series = Series(curve[2:]).astype('float')
    if as_series:
        return series
    return curve[:2] + series.tolist()

def plot_mortality_rate(curve, normalize=False, string = True):
    series = get_mortality_rate(curve, normalize, string, True)
    if series is not None:
        series.plot(label=curve[0])

def plot_all_mortality_rates(name, curves, threshold=25):
    entry = curves[name]
    for curve in entry[1:]:
        if threshold > 0 and curve.strip() != '' and int(curve[5:curve[5:].find(',') + 5]) < threshold:
            continue
        plot_mortality_rate(curve)
    plt.title('Mortality rates for ' + name.capitalize() + ' by birth year')
    plt.xlabel('Age')
    plt.ylabel('Rate per 1,000')
    plt.show()

def average_curves(curves, new_label='Average', normalize=1000):
    total_pop = 0
    entries = []
    for i in range(len(curves)):
        if curves[i].strip() != '':
            entry = curves[i].split(',')
            entries.append(entry)
            total_pop += int(entry[1])
    average = [new_label, total_pop]
    for age in range(120):
        died = 0
        pop = 0
        for curve in entries:
            if age + 2 < len(curve):
                pop += int(curve[1])
                died += int(curve[age+2])
        if pop > 0:
            average.append((died / pop) * 1000)
        else:  
            break
        #print('\rage %d' % age, end='')
    return average

#do not pass averages into this function
def normalize_to_mortality_rate(curve, string=False):
    if string:
        curve = curve.split(',')
    from pandas import Series
    series = Series(curve[2:]).astype('int') / int(curve[1]) * 1000
    result = series.tolist()
    result.insert(0, curve[1])
    result.insert(0, curve[0])
    return result

def score_curve(curve, baseline_curve, mode='add', strings=False, normalize1=False, normalize2=False, scale_score=False):
    if strings:
        curve = curve.split(',')
        baseline_curve = baseline_curve(',')
    score = ['mscore of ' + curve[0], int(curve[1]) + int(baseline_curve[1])]
    for i in range(2,122):
        if i >= len(curve) or i >= len(baseline_curve):
            break
        score.append(float(curve[i]) - float(baseline_curve[i]))
    #print('scored: %(len_c)d, baseline: %(len_bc)d, score: %(len_sc)d' % {"len_c":len(curve), "len_bc":len(baseline_curve), "len_sc":len(score)})
    if scale_score:
        import numpy as np
        scalar_curve = np.arange(10.0, 0., -.125)
        for i in range(2, len(score)):
            if i < len(scalar_curve):
                score[i] = score[i] * scalar_curve[i-2]
            else:
                score[i] = 0            
    score_num = -1
    if mode == 'add':
        # PROBLEM: this needs to reflect the fact that if the mortality rate is higher than average
        #early in life, the overall (numeric) mortality score should be high
        # the current SOLUTION to this is to only consider the mortality scores below age 80
        score_num = 0
        for item in score[2:]:
            score_num += item
        score_num /= 10
    if mode == 'avg':
        print('avg score not implemented')
    if mode == 'curve':
        return score
    return score_num

def plot_curve_avg_score(name, curves, show_score_curve=False):
    import matplotlib.pyplot as plt
    f = open('AVERAGE_MORTALITY_RATE','r')
    avg = []
    count = 0
    for line in f:
        if count > 1:
            avg.append(float(line))
        else:    
            avg.append(line)
        count += 1
    curve = average_curves(curves[name],new_label=name.capitalize())
    plot_mortality_rate(curve,string=False)
    plot_mortality_rate(avg,string=False)
    if show_score_curve:
        scurve = score_curve(curve, avg, mode='curve', scale_score=True)
        plot_mortality_rate(scurve,string=False)    
    plt.title('Mortality rate for ' + name + ' compared to baseline, mscore sum=%.2f'%score_curve(curve,avg,scale_score=True))
    plt.legend(bbox_to_anchor=(0.,.82,.75,.202), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def plot_curve_precomputed_avg_score(name, curves_avg, plot_peak=True):
    import matplotlib.pyplot as plt
    f = open('AVERAGE_MORTALITY_RATE','r')
    avg = []
    count = 0
    for line in f:
        if count > 1:
            avg.append(float(line))
        else:    
            avg.append(line)
        count += 1
    curve = curves_avg[name]
    curve[0] = name.capitalize()
    plot_mortality_rate(curve,string=False)
    plot_mortality_rate(avg,string=False)
    if plot_peak:
        peak = find_curve_peak(curve,string=False)
        plt.axvline(peak,color='k',linestyle='--')
        plt.title('Mortality rate for ' + name.capitalize() + ' compared to baseline, peak = %d'%peak)
    else:
        plt.title('Mortality rate for ' + name.capitalize() + ' compared to baseline, mscore sum=%.2f'%score_curve(curve,avg,scale_score=True))    
    plt.legend(bbox_to_anchor=(0.,.82,.75,.202), loc=3, ncol=2, mode="expand", borderaxespad=0.)    
    plt.show()
    
def get_mort_rate_averages_dictionary(curves):
    curves_avg = {}
    count = 0
    for name in curves.keys():
        curves_avg[name] = average_curves(curves[name], new_label=name)
        print('%.2f percent' % (100 * count / len(curves.keys())),end='\r')
        count += 1
    return curves_avg

def score_curves_dictionary(curves):
    f = open('AVERAGE_MORTALITY_RATE','r')
    avg = []
    count = 0
    for line in f:
        if count > 1:
            avg.append(float(line))
        else:    
            avg.append(line)
        count += 1
    count = 0
    scores = {}
    for name in curves.keys():
        scores[name] = score_curve(average_curves(curves[name]), avg, strings=False)
        count += 1
        print('\r%d scored' % count, end='')
    return scores

def score_curves_dictionary_by_decade(curves):
    curves_by_decade = {'1920':{},'1930':{},'1940':{},'1950':{},'1960':{},'1970':{},'1980':{},'1990':{},'2000':{}}
    for decade in curves_by_decade.keys():
        for name in curves.keys():
            curves_by_decade[decade][name] = []
            for curve in curves[name]:
                if curve.strip() != '' and int(curve[:4]) >= int(decade) and int(curve[:4]) < int(decade) + 10:
                    curves_by_decade[decade][name].append(curve) 
    scores_by_decade = {}
    for decade in curves_by_decade.keys():
        scores_by_decade[decade] = score_curves_dictionary(curves_by_decade[decade])
    return scores_by_decade

def count_population_curves_dictionary(curves):
    pop_dic = {}
    count = 0
    f = open('AVERAGE_MORTALITY_RATE','r')
    for name in curves.keys():
        pop = 0
        for curve in curves[name]:
            try:
                entry = curve.split(',')
            except:
                entry = curve
            for item in entry[2:]:
                pop += int(item)
        pop_dic[name] = pop
    return pop_dic

#TODO: discard? renormalize? all curves that exceed 1.0 fraction of deaths
def remove_contradictory_curves(curves,print_contradictions=False):
    curves_new = {}
    count=0
    total=0
    for name in curves.keys():
        curves_new[name] = []
        for i in range(len(curves[name])):
            if curves[name][i].strip() != '':                 
                entry = curves[name][i].split(',')
                births = int(entry[1])
                deaths = 0
                for died in entry[2:]:
                    deaths += int(died.strip())
                if deaths <= births:
                    curves_new[name].append(curves[name][i])
                elif print_contradictions:
                    print('FOUND CONTRADICTION: ' + name + ' ' + curves[name][i] + ' births:' + str(births) + ' deaths:' + str(deaths))
                    count += 1
            total += 1
    print('found %(count)d contradictions of %(total)d total curves, %(percent).2f percent' % {'count':count,'total':total,'percent':(100 * count / total)})
    return curves_new

def find_curve_peak(curve, string=True):
    age = 0
    highest_point = 0.
    if string:
        entry = curve.split(',')
    else:
        entry = curve
    for i in range(len(entry)):
        if i > 1 and float(entry[i]) > highest_point:
            age = i-2
            highest_point = float(entry[i])
    return age
