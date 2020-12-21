import numpy as np
import matplotlib.pyplot as plt
import copy

class ElectronicStatesParser:
    def __init__(self, doscar, eigfile):
        """
            Params: 
                - doscar : the location of DOSCAR file
                - eigfile: the location of EIGENVAL file -> needed for band structure
        """
        self.doscar = doscar
        self.eigfile = eigfile
        self.info = {"emax" : 0, "emin" : 0, "nedos" : 0, "fermi" : 0, "nelect": 0, "nkpoints": 0, "nbands": 0}
        self.parse_info()


    
    def parse_info(self):
        start_read = 6 
        start_info = 5
            
        with open(self.doscar, 'r') as file:
            contents = file.readlines()

        with open(self.eigfile, 'r') as file:
            lines = file.readlines()    
        
        itr = 0
        for key in self.info:
            if(itr < 4):
                self.info[key] = float( (contents[start_info].split())[itr] )
                itr += 1
            else:
                self.info[key] = int(lines[start_info].split()[itr-4])     
                itr += 1
        
    
    def parse_dos(self, pdos_flag = False, natom = [], tdos_flag = True, shift_to_fermi = True):
        """
            Input `pdos_flag = True` and specify the index of atoms according to POSCAR in `natom`
            
            Return: tdos, pdos
                - tdos = Total DOS
                - pdos = Projected DOS
        """
        
        dos_dict = {"energy": [], "dos_up": [], "dos_down": [], "intdos_up": [], "intdos_down" : []}
        pdos_dict = {"energy": [],  "s_up": [], "s_down": [], 
                     "py_up": [], "py_down" : [], "pz_up": [], "pz_down" : [], "px_up": [], "px_down" : [], 
                     "dxy_up": [], "dxy_down" : [], "dyz_up": [], "dyz_down" : [], "dz2r2_up": [], "dz2r2_down" : [],
                     "dxz_up": [], "dxz_down" : [], "dx2y2_up": [], "dx2y2_down" : []}
        tdos = dos_dict.copy()
        pdos = {}

        start_read = 6 
        start_info = 5


        with open(self.doscar, 'r') as file:
            contents = file.readlines()

        nion = int( (contents[0].split())[0] )


        if(tdos_flag == True):
            start = start_read
            end = start_read + self.info['nedos']
            idx = np.arange(start, end, step = 1, dtype=int)   
            for i in idx:
                dat = contents[i].split()

                if(shift_to_fermi == True):
                    dat[0] = float(dat[0]) - self.info['fermi']
                
                for itr in range(len(dat)):
                    key = list(tdos.keys())[itr]
                    
                    if(itr % 2 == 0 and itr != 0 and len(dat) != 3):
                        tdos[key].append( -1 * float(dat[itr]) )  # multiply by -1 for spin down
                    else: 
                        tdos[key].append( float(dat[itr]) ) 

        if(pdos_flag == True):
            for i in natom:
                dos_tmp = 0
                dat = 0
                dos_tmp = copy.deepcopy(pdos_dict)  #For not mutating the original template

                
                start = start_read + ( (i) * self.info['nedos'] )  + (i*1)
                end = start_read + ( (i+1) * self.info['nedos'] )  + (i*1)
                idx = np.arange(start, end, step = 1, dtype=int)

                for j in idx:
                    dat = contents[j].split()

                    if(shift_to_fermi == True):
                        dat[0] = float(dat[0]) - self.info['fermi']

                    
                    for itr in range(len(dat)):
                        key = list(dos_tmp.keys())[itr]
                        if(itr % 2 == 0 and itr != 0 and len(dat) != 3):
                            dos_tmp[key].append( -1 * float(dat[itr]) )
                        else:
                            dos_tmp[key].append( float(dat[itr]) )

                pdos[f'atom_{i}'] = dos_tmp

        return tdos, pdos
  

    def parse_band(self, shift_to_fermi = True):
        bands = {}
        kpoints = []

        start_info = 5
        start_read = start_info + 3

        with open(self.doscar, 'r') as file:
            lines = file.readlines()
            efermi = float(lines[5].split()[3])

        with open(self.eigfile, 'r') as file:
            lines = file.readlines()

        for band_idx in range(self.info['nbands']): 
            band = {"band_up": [], "band_down": []}

            for kpoint in range(self.info['nkpoints']):
                nline = start_read + (kpoint * self.info['nbands']) + band_idx + (2*kpoint)

                efermi_s = efermi if shift_to_fermi == True else 0

                band_up = float(lines[nline].split()[1])-efermi_s
                band_down = float(lines[nline].split()[2])-efermi_s
                band["band_up"].append(band_up)
                band["band_down"].append(band_down)

                if(band_idx == 0):
                    kpoints.append(lines[nline - 1].split())

            bands[f"band_{band_idx+1}"] = band

        return bands
    
    def get_spd(self, dat):
        """
            Sum up the projected dos on each coordinate to get the total s, p, and d orbitals DOS.
        """
        
        s = {'up': [], 'down': []}
        p = {'up': [], 'down': []}
        d = {'up': [], 'down': []}

        s['up'] = np.array(dat['s_up'])
        s['down'] = np.array(dat['s_down'])

        p['up'] = np.array(dat['py_up']) + np.array(dat['px_up']) + np.array(dat['pz_up'])
        p['down'] = np.array(dat['py_down']) + np.array(dat['px_down']) + np.array(dat['pz_down'])

        d['up'] = np.array(dat['dxy_up']) + np.array(dat['dyz_up']) + np.array(dat['dz2r2_up']) + np.array(dat['dxz_up']) + np.array(dat['dx2y2_up'])
        d['down'] = np.array(dat['dxy_down']) + np.array(dat['dyz_down']) + np.array(dat['dz2r2_down']) + np.array(dat['dxz_down']) + np.array(dat['dx2y2_down'])

        return s, p, d
    
    def get_kpoints(self, kpoint_file):
        start_read_k = 4
        with open(kpoint_file, 'r') as file:
            klines = file.readlines()

        nk = int(klines[1].split()[0])
        x_ticks = []
        x_ticks_label = []

        itr = 0
        tmp_arr = []
        kx=[] ; ky=[]; kz=[]

        for line in klines:
            if(itr > 0):  #skip the first row
                dat = line.split()

                if(len(dat) > 2):  #get the kpoints
                    if dat[-1] == 'GAMMA':
                        dat[-1] = '\Gamma'

                    dat[-1] = '$' + dat[-1] + '$'
                    tmp_arr.append(dat[-1])  

                    kx.append(float(dat[0]))
                    ky.append(float(dat[1]))
                    kz.append(float(dat[2]))          

            itr += 1

        dist_x = []
        dist_y = []
        dist_z = []
        dist = []

        for i in range(len(kx)-1):
            if(i%2 == 0):
                #print(i)
                dist_x.append(kx[i+1] - kx[i])
                dist_y.append(ky[i+1] - ky[i])
                dist_z.append(kz[i+1] - kz[i])

        dist = [np.sqrt(dist_x[i]**2 + dist_y[i]**2 + dist_z[i]**2) for i in range(len(dist_x))]
        dist[0] *= 0.3

        x_start = 0
        kpts_x = []
        for i in range(len(dist)):
            #print(x_start)
            x_ticks.append(x_start)

            one_path_x = np.linspace(x_start, x_start+dist[i],nk)
            kpts_x = np.hstack((kpts_x, one_path_x))
            x_start += dist[i]
        x_ticks.append(x_start)


        for label_idx in range(len(tmp_arr)):
            if(label_idx % 2 != 0 or label_idx == 0):
                x_ticks_label.append(tmp_arr[label_idx])

        return kpts_x, x_ticks, x_ticks_label
    
    def plot_band(self, bands, kpoints_file, give_ticks=True):
        kpts_x, x_ticks, x_ticks_label = self.get_kpoints(kpoints_file)
        
        for i in range(len(bands)):
            x = bands[f'band_{i+1}']['band_up']
            y = bands[f'band_{i+1}']['band_down']

            plt.plot(kpts_x, x, color='red')
            plt.plot(kpts_x, y, color='blue')

        if give_ticks == True:
            for i in range(len(x_ticks)):
                plt.axvline(x_ticks[i], linestyle='dotted', color='black')

        plt.xticks(x_ticks, x_ticks_label)
        plt.axis([0, kpts_x[-1], -2, 2])
        plt.ylabel(r"$E - E_f$ (eV)")
        plt.axhline(y = 0, color='black', linestyle='dashed')
        plt.yticks(np.arange(-1, 2.1, step = 1))