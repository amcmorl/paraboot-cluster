'''
paraboot_nullT_dispatcher
-------------------------

generate paraboot_nullT tasks using qsub and changing datasets, units and batch
numbers
'''
import paraboot_support as ps

# load dsnames, unit_names from file
unf = open(ps.unit_file, 'r')

#~ def gen_line(f):
    #~ f.seek(0, 2) # go to last
    #~ last_pos = unf.tell()
    #~ f.seek(0, 0) # go to start
    #~ while f.tell() < last_pos:
        #~ yield f.readline()

lines = unf.readlines()
for ibat in xrange(ps.nbat):
    for line in lines:
        dsname, unit_name = line.split()
        print "echo \"python paraboot_nullT %s %s %d --nrep=%d\" | qsub" % \
            (dsname, unit_name, ibat, ps.nrep_per_batch)
    
