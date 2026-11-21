from return_env import *
if __name__=='__main__':
    corrected_reference(RETURN_DIR);env=ReturnEnv();r=env.run(np.zeros(9),NOMINAL,record=True)
    (RETURN_DIR/'initial_rollout.json').write_text(json.dumps({'report':r,'frames':env.frames,'events':env.events}));print(json.dumps(r,indent=2),flush=True)
