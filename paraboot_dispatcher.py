'''
paraboot_nullT_dispatcher
-------------------------

generate paraboot_nullT tasks using qsub and changing datasets, units and batch
numbers
'''
import paraboot_support as ps

# load dsnames, unit_names from file
unf = open(ps.unit_file, 'r')

lines = unf.readlines()
for ibat in xrange(ps.nbat):
#for ibat in xrange(1):
    for line in lines:
        dsname, unit_name = line.split()
    	cmd_str = "python /data/code/paraboot_nullT.py %s %s %d --nrep=%d" % \
            (dsname, unit_name, ibat, ps.nrep_per_batch)
	with open(ps.job_dir + '/%s_%s_%d.sh' % (dsname, unit_name.lower(), ibat), 'w') as f:
	    f.write(cmd_str + '\n')
        print cmd_str
