mod buffer;
mod convert;
mod py_cmtm;
mod py_se3;
mod py_so3;

use py_cmtm::PyCmtm;
use py_se3::PySe3;
use py_so3::PySo3;
use pyo3::prelude::*;

#[pymodule]
pub fn mathrobors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySo3>()?;
    module.add_class::<PySe3>()?;
    module.add_class::<PyCmtm>()?;
    let cmtm = module.getattr("CMTM")?;
    module.setattr("SO3CMTM", cmtm.clone())?;
    module.setattr("SE3CMTM", cmtm)?;
    Ok(())
}
