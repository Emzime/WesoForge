pub const PROGRESS_BAR_STEPS: u64 = 20;

pub fn default_classgroup_element() -> [u8; 100] {
    let mut el = [0u8; 100];
    el[0] = 0x08;
    el
}
