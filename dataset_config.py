
DATASET_CONFIGS = {
  "D#1": {
    "data_path": "data/Archived/features_stress_fixed-current_D#1.pkl",
    "pra_threshold": 0.5,
    "auroc_threshold": 0.5,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": [
        "SCR_DUR", "SCR_EVENT", "APP_DUR_UNKNOWN", "APP_CAT", "BAT_LEV", "BAT_TMP", 
        "BAT_STA", "CHG", "RNG", "ONF", "PWS", "CON", "MED_IMG", "MED_VID", "MED_ALL"
      ],
      "Social": ["MSG_ALL", "MSG_RCV", "MSG_SNT", "CAE_CNT", "CAE_DUR"],
      "Physical": [
        "ACT", "STP", "HRT", "RRI", "EDA", "AML", "ACC_AXY", "ACC_AXX", "ACC_AXZ", "ACC_MAG",
        "ACE_WLK", "ACE_FOT", "ACE_UNK", "ACE_BCC", "ACE_RUN", "ACE_VHC", "ACE_TLT", "INS_JAC"
      ],
      "Mobility": [
        "LOC_DST", "LOC_LABEL", "WIF_EUC", "WIF_COS", "WIF_JAC", "WIF_MAN",
        "DAT_RCV", "DAT_SNT", "DST_PAC", "DST_MOT", "DST_SPD", "DST_DST",
        "ULV_INT", "ULV_EXP", "SKT"
      ],
      "Sleep": ["Sleep"]
    }
  },
  "D#4": {
    "data_path": "data/Archived/stress_binary_personal-current.pkl",
    "pra_threshold": 0.65,
    "auroc_threshold": 0.65,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": [
        "APP", "WLS", "CHG", "Dozemode", "PWR", "BAT_TMP", "SCR", "BAT", 
        "Notification", "RING", "Time", "ONOFF", "keyevent"
      ],
      "Social": ["CALL", "DATA_SNT", "DATA_RCV", "DATA_MRCV", "DATA_MSNT", "MSG_SNT", "MSG_RCV", "MSG_ALL"],
      "Physical": ["ACT", "FCL_VAL", "FAC_VAL", "FDI_VAL", "FST_VAL", "Fitbit", "ACE"],
      "Mobility": ["LOC", "INST_JAC", "WIFI_COS", "WIFI_EUC", "WIFI_MAN", "WIFI_JAC"],
      "Sleep": ["sleep", "Sleep"]
    }
  },
  "D#3": {
    "data_path": "data/Archived/stress_binary_personal-full_D#3.pkl",
    "pra_threshold": 0.65,
    "auroc_threshold": 0.65,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": [
        "SCR", "SCR_DUR", "SCR_EVENT", "APP_DUR_WORK", "APP_DUR_UNKNOWN", "APP_DUR_INFO", 
        "APP_DUR_HEALTH", "APP_DUR_SYSTEM", "APP_DUR_ENTER", "APP_DUR_SOCIAL", "APP_CAT", 
        "BAT_LEV", "BAT_PLG", "BAT_STA", "BAT_TMP", "CHG", "PWR", "Dozemode", "RING", 
        "Notification_CAT", "Notification_VIS", "keyevent_TIME", "keyevent_DIST", "keyevent_CAT"
      ],
      "Social": ["MSG_RCV", "MSG_ALL", "MSG_SNT", "CALL_CNT", "CALL_DUR"],
      "Physical": [
        "ACT", "ACE_WLK", "ACE_FOT", "ACE_UNK", "ACE_BCC", "ACE_RUN", "ACE_VHC", "ACE_TLT", 
        "INST_JAC", "FitbitStepcount", "FitbitHeartrate", "Fitbitdistance", "Fitbitcalorie"
      ],
      "Mobility": [
        "LOC_CLS", "LOC_DST", "LOC_LABEL", "WIFI_JAC", "WIFI_EUC", "WIFI_COS", "WIFI_MAN", 
        "BT_DeviceType", "BT_BondState", "BT_classType", "FDI_VAL", "FST_VAL", "FAC_VAL", 
        "FCL_VAL", "WLS", "DATA_RCV", "DATA_SNT", "DATA_MRCV", "DATA_MSNT"
      ],
      "Sleep": ["Sleep"]
    }
  },
  "crosscheck": {
    "data_path": "data/Archived_Tomiris/stress_binary_personal-current.pkl",
    "pra_threshold": 0.7,
    "auroc_threshold": 0.7,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": [
        "APP", "WLS", "CHG", "Dozemode", "PWR", "BAT_TMP", "SCR", "BAT", 
        "Notification", "RING", "Time", "ONOFF", "keyevent"
      ],
      "Social": [
        "CALL", "DATA_SNT", "DATA_RCV", "DATA_MRCV", "DATA_MSNT", 
        "MSG_SNT", "MSG_RCV", "MSG_ALL", "CAL", "MSG", "CON"
      ],
      "Physical": ["ACT", "FCL_VAL", "FAC_VAL", "FDI_VAL", "FST_VAL", "Fitbit", "ACE", "TOTAL"],
      "Mobility": ["LOC_", "INST_JAC", "WIFI_COS", "WIFI_EUC", "WIFI_MAN", "WIFI_JAC"],
      "Sleep": ["sleep", "Sleep", "SLP"]
    }
  },
  "studentlife": {
    "data_path": "data/Archived_Tomiris/stress_binary_personal_sl.pkl",
    "pra_threshold": 0.5,
    "auroc_threshold": 0.5,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": ["SCR_DUR"],
      "Social": ["CON_CNT", "CON_DUR"],
      "Physical": ["ACT_STILL", "ACT_WALKING", "ACT_UNKNOWN", "ACT_RUNNING", "ACT_FOOT", "TOTAL_ACT"],
      "Mobility": ["LOC_CNT", "LOC_DST", "LOC_DST_PER_PLACE"],
      "Sleep": ["SLP_START", "SLP_END", "SLP_DUR"]
    }
  },
  "D#2": {
    "data_path": "data/Archived/stress_binary_personal-full_D#2.pkl",
    "pra_threshold": 0.6,
    "auroc_threshold": 0.65,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "Sleep": ["Sleep"],
      "Physical": [
        "ACT", "FitbitStepcount", "Fitbitcalorie", "Fitbitdistance", "FitbitHeartrate", 
        "ACE_FOT", "ACE_TLT", "ACE_WLK", "ACE_BCC", "ACE_VHC", "ACE_UNK", "ACE_RUN"
      ],
      "PhoneUsage": [
        "SCR_EVENT", "SCR", "SCR_DUR", "BAT_STA", "BAT_LEV", "BAT_PLG", "BAT_TMP", "PWR", 
        "Dozemode", "ONOFF", "CHG", "APP_DUR_WORK", "APP_DUR_UNKNOWN", "APP_DUR_SOCIAL", 
        "APP_DUR_ENTER", "APP_DUR_SYSTEM", "APP_DUR_HEALTH", "APP_DUR_INFO", "APP_CAT", 
        "RING", "Notification_VIS", "Notification_CAT", "Time", "WLS"
      ],
      "Social": ["CALL_DUR", "CALL_CNT", "MSG_SNT", "MSG_RCV", "MSG_ALL"],
      "Mobility": [
        "LOC_DST", "LOC_CLS", "LOC_LABEL", "DATA_SNT", "DATA_RCV", "DATA_MRCV", 
        "DATA_MSNT", "WIFI_JAC", "WIFI_COS", "WIFI_EUC", "WIFI_MAN", "INST_JAC", "BT_RSSI"
      ]
    }
  },
  "GLOBEM": {
    "data_path": "RQ1/data/Overfitting/GLOBEM/Intermediate/depression_globem_combined_no_missing.pkl",
    "pra_threshold": 0.5,
    "auroc_threshold": 0.5,
    "top_k": 40,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "phone_usage": ["phone_usage", "keyevent", "APP", "SCR"],
      "mobility": ["mobility", "LOC"],
      "physical_status": [
        "physical", "ACT", "FCL", "FAC", "FDI", "FST", "FitbitHeartrate",
        "FitbitStepcount", "Fitbitcalorie", "Fitbitdistance"
      ],
      "sleep": ["sleep"],
      "social_behavior": ["social", "CALL", "MSG"]
    }
  },
  "step_count": {
    "data_path": "data/Archived/step_count_binary_personal-15min.pkl",
    "pra_threshold": 0.8,
    "auroc_threshold": 0.85,
    "top_k": 3,
    "concept_shift_k": 2,
    "concept_shift_min_samples": 20,
    "category_prefixes": {
      "PhoneUsage": [],
      "Social": [],
      "Physical": [
        "CALL_DUR", "CALL_CNT", "DATA_RCV", "DATA_SNT", "DATA_MRCV", "DATA_MSNT", "MSG_SNT", 
        "MSG_RCV", "MSG_ALL", "CAL", "Heartrate", "LOC_LABEL", "LOC_DST", "APP_DUR_UNKNOWN", 
        "APP_CAT", "BAT_LEV", "BAT_STA", "BAT_TMP", "BAT_PLG", "SCR_EVENT", "SCR_DUR", 
        "RING", "CHG", "Dozemode", "Notification_VIS", "Notification_CAT", "Time", "PWR", 
        "INST_JAC", "BT_BondState", "BT_DeviceType", "BT_classType"
      ],
      "Mobility": [],
      "Sleep": ["Sleep"]
    }
  }
}
